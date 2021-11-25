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

import com.intel.oap.GazellePluginConfig
import com.intel.oap.expression.ConverterUtils
import com.intel.oap.vectorized.{BatchIterator, _}
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.spark._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.connector.read.{InputPartition, PartitionReaderFactory}
import org.apache.spark.sql.execution.datasources.FilePartition
import org.apache.spark.sql.execution.datasources.v2.VectorizedFilePartitionReaderHandler
import org.apache.spark.sql.execution.datasources.v2.arrow.SparkMemoryUtils
import org.apache.spark.sql.execution.datasources.v2.parquet.ParquetPartitionReaderFactory
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.sql.util.OASPackageBridge._
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}
import org.apache.spark.util._

class WholestageRDDPartition(val index: Int, val inputPartition: InputPartition)
  extends Partition
    with Serializable

class WholestageColumnarRDD(
    sc: SparkContext,
    @transient private val inputPartitions: Seq[InputPartition],
    partitionReaderFactory: PartitionReaderFactory,
    columnarReads: Boolean,
    batchIter: BatchIterator,
    output: Seq[Attribute],
    tmp_dir: String)
    extends RDD[ColumnarBatch](sc, Nil) {
  val numaBindingInfo = GazellePluginConfig.getConf.numaBindingInfo

  override protected def getPartitions: Array[Partition] = {
    inputPartitions.zipWithIndex.map {
      case (inputPartition, index) => new WholestageRDDPartition(index, inputPartition)
    }.toArray
  }

  private def castPartition(split: Partition): WholestageRDDPartition = split match {
    case p: WholestageRDDPartition => p
    case _ => throw new SparkException(s"[BUG] Not a DataSourceRDDPartition: $split")
  }

  override def compute(split: Partition, context: TaskContext): Iterator[ColumnarBatch] = {
    ExecutorManager.tryTaskSet(numaBindingInfo)
    val outputSchema = ConverterUtils.toArrowSchema(output)
//    val inputPartition = castPartition(split).inputPartition
//    inputPartition match {
//      case p: FilePartition =>
//        p.files.foreach { f => inputSize += f.length }
//      case _ =>
//    }

    val iter = new Iterator[Any] {
      private val inputMetrics = TaskContext.get().taskMetrics().inputMetrics

      private[this] var valuePrepared = false

      override def hasNext: Boolean = {
        val res = batchIter.hasNext
        res
      }

      override def next(): Any = {
        if (!hasNext) {
          throw new java.util.NoSuchElementException("End of stream")
        }
        val rb = batchIter.next()
        if (rb == null) {
          val resultStructType = ArrowUtils.fromArrowSchema(outputSchema)
          val resultColumnVectors =
            ArrowWritableColumnVector.allocateColumns(0, resultStructType).toArray
          return new ColumnarBatch(resultColumnVectors.map(_.asInstanceOf[ColumnVector]), 0)
        }
        val outputNumRows = rb.getLength
        val output = ConverterUtils.fromArrowRecordBatch(outputSchema, rb)
        ConverterUtils.releaseArrowRecordBatch(rb)
        val cb = new ColumnarBatch(output.map(v => v.asInstanceOf[ColumnVector]), outputNumRows)
        val bytes: Long = cb match {
          case batch: ColumnarBatch =>
            (0 until batch.numCols()).map { i =>
              val vector = Option(batch.column(i))
              vector.map {
                case av: ArrowWritableColumnVector =>
                  av.getValueVector.getBufferSize.toLong
                case _ => 0L
              }.sum
            }.sum
          case _ => 0L
        }
        inputMetrics.bridgeIncBytesRead(bytes)
        cb
      }
    }
    val closeableColumnarBatchIterator = new CloseableColumnBatchIterator(
      iter.asInstanceOf[Iterator[ColumnarBatch]])
    // TODO: SPARK-25083 remove the type erasure hack in data source scan
    new InterruptibleIterator(context, closeableColumnarBatchIterator)
  }

  override def getPreferredLocations(split: Partition): Seq[String] = {
    castPartition(split).inputPartition.preferredLocations()
  }

}
