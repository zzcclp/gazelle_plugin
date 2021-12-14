/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.oap.execution

import scala.collection.mutable.ListBuffer

import com.google.common.collect.Lists
import com.intel.oap.GazellePluginConfig
import com.intel.oap.substrait.extensions.{MappingBuilder, MappingNode}
import com.intel.oap.substrait.plan.PlanBuilder
import com.intel.oap.vectorized._
import org.apache.spark._

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.UnsafeRow
import org.apache.spark.sql.connector.read.{InputPartition, PartitionReaderFactory}
import org.apache.spark.sql.execution.arrow.{CHBatchIterator, EmptyBatchIterator}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.datasources.FilePartition
import org.apache.spark.util._

class WholestageClickhouseRowRDD(
    sc: SparkContext,
    @transient private val inputPartitions: Seq[InputPartition],
    partitionReaderFactory: PartitionReaderFactory,
    columnarReads: Boolean,
    lastPlanInWS: SparkPlan,
    jarList: Seq[String],
    dependentKernelIterators: ListBuffer[BatchIterator],
    tmp_dir: String)
    extends RDD[InternalRow](sc, Nil) {
  val numaBindingInfo = GazellePluginConfig.getConf.numaBindingInfo

  override protected def getPartitions: Array[Partition] = {
    inputPartitions.zipWithIndex.map {
      case (inputPartition, index) => new WholestageRDDPartition(index, inputPartition)
    }.toArray
  }

  private def castPartition(split: Partition): WholestageRDDPartition = split match {
    case p: WholestageRDDPartition => p
    case _ => throw new SparkException(s"[BUG] Not a WholestageRDDPartition: $split")
  }

  private def doWholestageTransform(index: java.lang.Integer,
                                    paths: java.util.ArrayList[String],
                                    starts: java.util.ArrayList[java.lang.Long],
                                    lengths: java.util.ArrayList[java.lang.Long])
    : WholestageTransformContext = {
    val functionMap = new java.util.HashMap[String, java.lang.Long]()
    val childCtx = lastPlanInWS.asInstanceOf[TransformSupport]
      .doTransform(functionMap, index, paths, starts, lengths)
    if (childCtx == null) {
      throw new NullPointerException(
        s"ColumnarWholestageTransformer can't doTansform on ${lastPlanInWS}")
    }
    val mappingNodes = new java.util.ArrayList[MappingNode]()
    val mapIter = functionMap.entrySet().iterator()
    while(mapIter.hasNext) {
      val entry = mapIter.next()
      val mappingNode = MappingBuilder.makeFunctionMapping(entry.getKey, entry.getValue)
      mappingNodes.add(mappingNode)
    }
    val relNodes = Lists.newArrayList(childCtx.root)
    val planNode = PlanBuilder.makePlan(mappingNodes, relNodes)
    WholestageTransformContext(childCtx.inputAttributes,
      childCtx.outputAttributes, planNode)
  }

  override def compute(split: Partition, context: TaskContext): Iterator[InternalRow] = {
    ExecutorManager.tryTaskSet(numaBindingInfo)

    var index: java.lang.Integer = null
    val paths = new java.util.ArrayList[String]()
    val starts = new java.util.ArrayList[java.lang.Long]()
    val lengths = new java.util.ArrayList[java.lang.Long]()
    val inputPartition = castPartition(split).inputPartition
    inputPartition match {
      case p: FilePartition =>
        index = new java.lang.Integer(p.index)
        p.files.foreach { f =>
          paths.add(f.filePath.substring(7))
          starts.add(new java.lang.Long(f.start))
          lengths.add(new java.lang.Long(f.length))}
      case other =>
        throw new UnsupportedOperationException(s"$other is not supported yet.")
    }

    var startTime = System.nanoTime()
    val wsCtx = doWholestageTransform(index, paths, starts, lengths)
    logWarning(s"===========1 ${System.nanoTime() - startTime}")
    startTime = System.nanoTime()

    logWarning(s"Substrait Plan:\n${wsCtx.root.toProtobuf.toString}")
    val resIter = if (context.getLocalProperty("spark.oap.sql.columnar.use.emptyiter").toBoolean) {
      new EmptyBatchIterator()
    } else {
      new CHBatchIterator(wsCtx.root.toProtobuf.toByteArray, context.getLocalProperty("spark.oap.sql.columnar.ch.so.filepath"))
    }
    logWarning(s"===========2 ${System.nanoTime() - startTime}")

    val iter = new Iterator[InternalRow] with AutoCloseable {
      private val inputMetrics = TaskContext.get().taskMetrics().inputMetrics
      private val numFields = wsCtx.outputAttributes.length
      private[this] var currentIterator: Iterator[InternalRow] = null
      private var totalBatch = 0

      override def hasNext: Boolean = {
        val hasNextRes = (currentIterator != null && currentIterator.hasNext) || nextIterator()
        hasNextRes
      }

      private def nextIterator(): Boolean = {
        if (resIter.hasNext) {
          val startTime = System.nanoTime()
          val sparkRowInfo = resIter.next()
          totalBatch += 1
          logWarning(s"===========3 ${System.nanoTime() - startTime}")
          val result = if (sparkRowInfo.offsets != null && sparkRowInfo.offsets.length > 0) {
            val numRows = sparkRowInfo.offsets.length
            currentIterator = new Iterator[InternalRow] with AutoCloseable {

              var rowId = 0
              val row = new UnsafeRow(numFields)

              override def hasNext: Boolean = {
                rowId < numRows
              }

              override def next(): InternalRow = {
                if (rowId >= numRows) throw new NoSuchElementException
                val (offset, length) = (sparkRowInfo.offsets(rowId), sparkRowInfo.lengths(rowId))
                row.pointTo(null, sparkRowInfo.memoryAddress + offset, length.toInt)
                rowId += 1
                row
              }

              override def close(): Unit = {}
            }
            true
          } else {
            false
          }
          result
        } else {
          false
        }
      }

      override def next(): InternalRow = {
        if (!hasNext) {
          throw new java.util.NoSuchElementException("End of stream")
        }
        val cb = currentIterator.next()
        cb
      }

      override def close(): Unit = {
        resIter.close()
      }
    }
    iter
  }

  override def getPreferredLocations(split: Partition): Seq[String] = {
    castPartition(split).inputPartition.preferredLocations()
  }

}
