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

package org.apache.spark.sql.sources

import org.apache.spark.SparkConf
import org.apache.spark.sql.AnalysisException
import org.apache.spark.sql.catalyst.util.DateTimeUtils
import org.apache.spark.sql.execution.datasources.DataSource
import org.apache.spark.sql.test.SharedSparkSession

class ResolvedDataSourceSuite extends SharedSparkSession {

  override def sparkConf: SparkConf =
    super.sparkConf
      .setAppName("test")
      .set("spark.sql.parquet.columnarReaderBatchSize", "4096")
      .set("spark.sql.sources.useV1SourceList", "avro")
      .set("spark.sql.extensions", "com.intel.oap.ColumnarPlugin")
      .set("spark.sql.execution.arrow.maxRecordsPerBatch", "4096")
      //.set("spark.shuffle.manager", "org.apache.spark.shuffle.sort.ColumnarShuffleManager")
      .set("spark.memory.offHeap.enabled", "true")
      .set("spark.memory.offHeap.size", "50m")
      .set("spark.sql.join.preferSortMergeJoin", "false")
      .set("spark.unsafe.exceptionOnMemoryLeak", "false")
      //.set("spark.oap.sql.columnar.tmp_dir", "/codegen/nativesql/")
      .set("spark.sql.columnar.sort.broadcastJoin", "true")
      .set("spark.oap.sql.columnar.preferColumnar", "true")
      .set("spark.oap.sql.columnar.sortmergejoin", "true")

  private def getProvidingClass(name: String): Class[_] =
    DataSource(
      sparkSession = spark,
      className = name,
      options = Map(DateTimeUtils.TIMEZONE_OPTION -> DateTimeUtils.defaultTimeZone().getID)
    ).providingClass

  test("jdbc") {
    assert(
      getProvidingClass("jdbc") ===
      classOf[org.apache.spark.sql.execution.datasources.jdbc.JdbcRelationProvider])
    assert(
      getProvidingClass("org.apache.spark.sql.execution.datasources.jdbc") ===
      classOf[org.apache.spark.sql.execution.datasources.jdbc.JdbcRelationProvider])
    assert(
      getProvidingClass("org.apache.spark.sql.jdbc") ===
        classOf[org.apache.spark.sql.execution.datasources.jdbc.JdbcRelationProvider])
  }

  test("json") {
    assert(
      getProvidingClass("json") ===
      classOf[org.apache.spark.sql.execution.datasources.json.JsonFileFormat])
    assert(
      getProvidingClass("org.apache.spark.sql.execution.datasources.json") ===
        classOf[org.apache.spark.sql.execution.datasources.json.JsonFileFormat])
    assert(
      getProvidingClass("org.apache.spark.sql.json") ===
        classOf[org.apache.spark.sql.execution.datasources.json.JsonFileFormat])
  }

  test("parquet") {
    assert(
      getProvidingClass("parquet") ===
      classOf[org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat])
    assert(
      getProvidingClass("org.apache.spark.sql.execution.datasources.parquet") ===
        classOf[org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat])
    assert(
      getProvidingClass("org.apache.spark.sql.parquet") ===
        classOf[org.apache.spark.sql.execution.datasources.parquet.ParquetFileFormat])
  }

  test("csv") {
    assert(
      getProvidingClass("csv") ===
        classOf[org.apache.spark.sql.execution.datasources.csv.CSVFileFormat])
    assert(
      getProvidingClass("com.databricks.spark.csv") ===
        classOf[org.apache.spark.sql.execution.datasources.csv.CSVFileFormat])
  }

  test("avro: show deploy guide for loading the external avro module") {
    Seq("avro", "org.apache.spark.sql.avro").foreach { provider =>
      val message = intercept[AnalysisException] {
        getProvidingClass(provider)
      }.getMessage
      assert(message.contains(s"Failed to find data source: $provider"))
      assert(message.contains("Please deploy the application as per the deployment section of"))
    }
  }

  test("kafka: show deploy guide for loading the external kafka module") {
    val message = intercept[AnalysisException] {
      getProvidingClass("kafka")
    }.getMessage
    assert(message.contains("Failed to find data source: kafka"))
    assert(message.contains("Please deploy the application as per the deployment section of"))
  }

  test("error message for unknown data sources") {
    val error = intercept[ClassNotFoundException] {
      getProvidingClass("asfdwefasdfasdf")
    }
    assert(error.getMessage.contains("Failed to find data source: asfdwefasdfasdf."))
  }
}