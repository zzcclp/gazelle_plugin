package com.intel.oap.benchmarks

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

import java.io.File

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

import org.apache.spark.sql.SparkSession

object BenchmarkTest {

  def main(args: Array[String]): Unit = {

    val (parquetFilesPath, fileFormat,
    executedCnt, configed, sqlFilePath, stopFlagFile) = if (args.length > 0) {
      (args(0), args(1), args(2).toInt, true, args(3), args(4))
    } else {
      val rootPath = this.getClass.getResource("/").getPath
      /* (new File(rootPath + "../../../../../tpch-data")
        .getAbsolutePath, "arrow", 10, false,
        rootPath + "queries/q06.sql", "") */
      ("/data1/test_output/tpch-data-sf10-nonesnappy", "parquet", 30, false,
        rootPath + "queries/q06.sql", "")
    }

    val sqlStr = Source.fromFile(new File(sqlFilePath), "UTF-8")

    val sessionBuilderTmp = SparkSession
      .builder()
      .appName("CH-As-Lib-Benchmark")

    val sessionBuilder = if (!configed) {
      sessionBuilderTmp
        .master("local[3]")
        .config("spark.driver.memory", "4G")
        .config("spark.driver.memoryOverhead", "6G")
        //.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
        //.config("spark.sql.warehouse.dir", warehouse)
        .config("spark.default.parallelism", 1)
        .config("spark.sql.shuffle.partitions", 1)
        .config("spark.sql.adaptive.enabled", "false")
        //.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        //.config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "2")
        //.config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "2MB")
        .config("spark.sql.files.maxPartitionBytes", 1536 << 10 << 10) // default is 128M
        .config("spark.sql.files.minPartitionNum", "1")
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.locality.wait", "0s")
        //.config("spark.sql.sources.ignoreDataLocality", "false")
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        //.config("spark.hadoop.parquet.block.size", "8388608")
        //.config("spark.sql.crossJoin.enabled", "true")
        //.config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.sources.useV1SourceList", "avro")
        .config("spark.memory.fraction", "0.3")
        .config("spark.memory.storageFraction", "0.3")
        //.config("spark.sql.parquet.columnarReaderBatchSize", "20000")
        //.config("spark.plugins", "com.intel.oap.GazellePlugin")
        //.config("spark.sql.execution.arrow.maxRecordsPerBatch", "20000")
        //.config("spark.oap.sql.columnar.columnartorow", "false")
        //.config("spark.oap.sql.columnar.use.emptyiter", "false")
        //.config("spark.oap.sql.columnar.ch.so.filepath",
        //  "/home/myubuntu/Works/c_cpp_projects/Kyligence-ClickHouse/cmake-build-debug/utils/local-engine/liblocal_engine_jnid.so")
        .config("spark.oap.sql.columnar.ch.so.filepath",
          "/home/myubuntu/Works/c_cpp_projects/Kyligence-ClickHouse/cmake-build-release/utils/local-engine/liblocal_engine_jni.so")
        //.config("spark.sql.planChangeLog.level", "info")
        .config("spark.sql.columnVector.offheap.enabled", "true")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "6442450944")
      //.config("javax.jdo.option.ConnectionURL", s"jdbc:derby:;databaseName=$hiveMetaStoreDB;" +
      //  s"create=true")
      //.enableHiveSupport()
    } else {
      sessionBuilderTmp
    }

    val spark = sessionBuilder.getOrCreate()
    if (!configed) {
      spark.sparkContext.setLogLevel("WARN")
    }

    testSQL(spark, parquetFilesPath, fileFormat, executedCnt, sqlStr.mkString)

    System.out.println("waiting for finishing")
    if (stopFlagFile.isEmpty) {
      Thread.sleep(1800000)
    } else {
      while ((new File(stopFlagFile)).exists()) {
        Thread.sleep(1000)
      }
    }
    spark.stop()
    System.out.println("finished")
  }

  def testSQL(spark: SparkSession, parquetFilesPath: String,
              fileFormat: String, executedCnt: Int,
              sql: String): Unit = {
    createTempView(spark, parquetFilesPath, fileFormat)

    val tookTimeArr = ArrayBuffer[Long]()
    for (i <- 1 to executedCnt) {
      val startTime = System.nanoTime()
      //spark.sql(sql).show(200, false)
      spark.sql(
        """
          | SELECT
          |    sum(l_extendedprice * l_discount) AS revenue
          | FROM
          |    lineitem
          | WHERE
          |    l_shipdate >= date'1994-01-01'
          |    AND l_shipdate < date'1994-01-01' + interval 1 year
          |    AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01
          |    AND l_quantity < 24;
          |""".stripMargin).show(200, false)
      /*spark.sql(
        """
          | SELECT
          |    sum(l_extendedprice * l_discount) AS revenue
          | FROM
          |    lineitem
          | WHERE
          |    l_returnflag = 'A' AND l_linestatus = 'F'
          |    AND l_shipdate >= date'1994-01-01'
          |    AND l_shipdate < date'1994-01-01' + interval 1 year
          |    AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01
          |    AND l_quantity < 24;
          |""".stripMargin).show(200, false)*/
      /* spark.sql(
        """
          | SELECT sum(l_orderkey), sum(l_partkey), sum(l_suppkey), sum(l_linenumber), sum(l_quantity), sum(l_extendedprice), sum(l_discount), sum(l_tax)
          | FROM lineitem;
          |""".stripMargin).show(10, false) */
      val tookTime = (System.nanoTime() - startTime) / 1000000
      println(s"Execute ${i} time, time: ${tookTime}")
      tookTimeArr += tookTime
    }

    println(tookTimeArr.mkString(","))

    spark.conf.set("org.apache.spark.example.columnar.enabled", "false")
    import spark.implicits._
    val df = spark.sparkContext.parallelize(tookTimeArr.toSeq, 1).toDF("time")
    df.summary().show(100, false)
  }

  def createTempView(spark: SparkSession, parquetFilesPath: String, fileFormat: String): Unit = {
    val dataSourceMap = Map(
      "customer" -> spark.read.format(fileFormat).load(parquetFilesPath + "/customer"),

      "lineitem" -> spark.read.format(fileFormat).load(parquetFilesPath + "/lineitem"),

      "nation" -> spark.read.format(fileFormat).load(parquetFilesPath + "/nation"),

      "region" -> spark.read.format(fileFormat).load(parquetFilesPath + "/region"),

      "orders" -> spark.read.format(fileFormat).load(parquetFilesPath + "/order"),

      "part" -> spark.read.format(fileFormat).load(parquetFilesPath + "/part"),

      "partsupp" -> spark.read.format(fileFormat).load(parquetFilesPath + "/partsupp"),

      "supplier" -> spark.read.format(fileFormat).load(parquetFilesPath + "/supplier"))

    dataSourceMap.foreach {
      case (key, value) => value.createOrReplaceTempView(key)
    }
  }
}
