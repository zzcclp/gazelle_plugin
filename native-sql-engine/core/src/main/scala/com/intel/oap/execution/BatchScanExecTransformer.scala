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
import com.intel.oap.substrait.rel.RelBuilder
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.connector.read.{InputPartition, PartitionReaderFactory, Scan}
import org.apache.spark.sql.execution.SparkPlan
import org.apache.spark.sql.execution.datasources.v2.BatchScanExec
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}

class BatchScanExecTransformer(output: Seq[AttributeReference], @transient scan: Scan)
    extends BatchScanExec(output, scan) with TransformSupport {
  val tmpDir: String = GazellePluginConfig.getConf.tmpFile
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
    var RDDs = Seq[RDD[ColumnarBatch]]()
    RDDs = RDDs :+ doExecuteColumnar()
    RDDs
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
    val typeNodes = ConverterUtils.getTypeNodeFromAttributes(output)
    val nameList = new java.util.ArrayList[String]()
    for (attr <- output) {
      nameList.add(attr.name)
    }
    val pathList = new java.util.ArrayList[String]()
    val relNode = RelBuilder.makeReadRel(typeNodes, nameList, pathList)
    val outputSchema = ConverterUtils.toArrowSchema(output)
    // FIXME
    val inputSchema = ConverterUtils.toArrowSchema(output)
    TransformContext(inputSchema, outputSchema, relNode)
  }
}
