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

import com.google.common.collect.Lists
import com.intel.oap.GazellePluginConfig
import com.intel.oap.expression.{ConverterUtils, ExpressionConverter, ExpressionTransformer}
import com.intel.oap.substrait.rel.{LocalFilesBuilder, RelBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.connector.read.{InputPartition, PartitionReaderFactory, Scan}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.datasources.v2.BatchScanExec
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}
import com.intel.oap.spark.sql.execution.datasources.v2.arrow.ArrowScan
import com.intel.oap.substrait.expression.ExpressionNode
import org.apache.spark.sql.execution.datasources.FilePartition

class BatchScanExecTransformer(output: Seq[AttributeReference], @transient scan: Scan)
    extends BatchScanExec(output, scan) with TransformSupport {
  val tmpDir: String = GazellePluginConfig.getConf.tmpFile
  val filterExprs: Seq[Expression] = scan.asInstanceOf[ArrowScan].dataFilters
  override def supportsColumnar(): Boolean = true
  override lazy val metrics = Map(
    "numOutputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "numInputBatches" -> SQLMetrics.createMetric(sparkContext, "input_batches"),
    "numOutputBatches" -> SQLMetrics.createMetric(sparkContext, "output_batches"),
    "scanTime" -> SQLMetrics.createTimingMetric(sparkContext, "totaltime_batchscan"),
    "inputSize" -> SQLMetrics.createSizeMetric(sparkContext, "input size in bytes"))

  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val numOutputRows = longMetric("numOutputRows")
    val numInputBatches = longMetric("numInputBatches")
    val numOutputBatches = longMetric("numOutputBatches")
    val scanTime = longMetric("scanTime")
    val inputSize = longMetric("inputSize")
    val inputColumnarRDD =
      new ColumnarDataSourceRDD(sparkContext, partitions, readerFactory, true, scanTime, numInputBatches, inputSize, tmpDir)
    inputColumnarRDD.map { r =>
      numOutputRows += r.numRows()
      numOutputBatches += 1
      r
    }
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[BatchScanExecTransformer]

  override def equals(other: Any): Boolean = other match {
    case that: BatchScanExecTransformer =>
      (that canEqual this) && super.equals(that)
    case _ => false
  }

  override def columnarInputRDDs: Seq[RDD[ColumnarBatch]] = {
    null
  }

  override def getBuildPlans: Seq[(SparkPlan, SparkPlan)] = {
    Seq((this, null))
  }

  override def getStreamedLeafPlan: SparkPlan = {
    this
  }

  override def getChild: SparkPlan = {
    null
  }

  override def doValidate(): Boolean = false

  override def doTransform(args: Object): TransformContext = {
    throw new UnsupportedOperationException(s"This operator doesn't support doTransform.")
  }

  override def doTransform(args: java.lang.Object,
                           index: java.lang.Integer,
                           paths: java.util.ArrayList[String],
                           starts: java.util.ArrayList[java.lang.Long],
                           lengths: java.util.ArrayList[java.lang.Long]): TransformContext = {
    val typeNodes = ConverterUtils.getTypeNodeFromAttributes(output)
    val nameList = new java.util.ArrayList[String]()
    for (attr <- output) {
      nameList.add(attr.name)
    }
    val filterNodes = filterExprs.toList.map(expr => {
      val transformer = ExpressionConverter.replaceWithExpressionTransformer(expr, output)
      transformer.asInstanceOf[ExpressionTransformer].doTransform(args)
    })
    val filterList = new java.util.ArrayList[ExpressionNode]()
    for (filterNode <- filterNodes) {
      filterList.add(filterNode)
    }
    val partNode = LocalFilesBuilder.makeLocalFiles(index, paths, starts, lengths)
    val relNode = RelBuilder.makeReadRel(typeNodes, nameList, filterList, partNode)
    TransformContext(output, output, relNode)
  }
}
