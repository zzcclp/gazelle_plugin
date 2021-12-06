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

import org.apache.spark.sql.SparkSession

object TPCHDemo {

  def main(args: Array[String]): Unit = {
    val sessionBuilder = SparkSession
      .builder()
      .master("local[4]")
      .appName("TPCH-Demo")
      .config("spark.driver.memory", "4G")
      .config("spark.driver.memoryOverhead", "6G")
      .config("spark.default.parallelism", 4)
      .config("spark.sql.shuffle.partitions", 4)
      .config("spark.sql.adaptive.enabled", "false")
      .config("spark.sql.files.maxPartitionBytes", 64 << 10 << 10)  // default is 128M
      .config("spark.sql.files.minPartitionNum", "1")
      .config("spark.sql.parquet.filterPushdown", "true")
      .config("spark.locality.wait", "0s")
      .config("spark.sql.sources.ignoreDataLocality", "true")
      .config("spark.sql.parquet.enableVectorizedReader", "true")
      //.config("spark.hadoop.parquet.block.size", "8388608")
      .config("spark.sql.sources.useV1SourceList", "avro")
      .config("spark.memory.fraction", "0.3")
      .config("spark.memory.storageFraction", "0.3")
      .config("spark.sql.parquet.columnarReaderBatchSize", "20000")
      .config("spark.sql.planChangeLog.level", "info")
      .config("spark.sql.columnVector.offheap.enabled", "true")
      .config("spark.memory.offHeap.enabled", "true")
      .config("spark.memory.offHeap.size", "6442450944")

    val spark = sessionBuilder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    testTPCH(spark)

    System.out.println("waiting for finishing")
    Thread.sleep(1800000)
    spark.stop()
    System.out.println("finished")
  }

  def testTPCH(spark: SparkSession): Unit = {

    val currentPath = this.getClass.getResource("/").getPath
    val queriesPath = currentPath + "queries/"
    val parquetFilesPath = new File(currentPath + "../../../../../tpch-data")
      .getAbsolutePath
    val dataSourceMap = Map(
      "customer" -> spark.read.parquet(parquetFilesPath + "/customer"),

      "lineitem" -> spark.read.parquet(parquetFilesPath + "/lineitem"),

      "nation" -> spark.read.parquet(parquetFilesPath + "/nation"),

      "region" -> spark.read.parquet(parquetFilesPath + "/region"),

      "orders" -> spark.read.parquet(parquetFilesPath + "/order"),

      "part" -> spark.read.parquet(parquetFilesPath + "/part"),

      "partsupp" -> spark.read.parquet(parquetFilesPath + "/partsupp"),

      "supplier" -> spark.read.parquet(parquetFilesPath + "/supplier"))

    dataSourceMap.foreach {
      case (key, value) => value.createOrReplaceTempView(key)
    }
    /*spark.sql(
      """
        | show tables;
        |""".stripMargin).show(200, false)*/

    import scala.io.Source

    for (i <- 1 to 22) {
      val queryId = "q%02d.sql".format(i)
      val source = Source.fromFile(new File(queriesPath + queryId), "UTF-8")
      spark.sparkContext.setJobDescription(s"Query ${queryId}")
      spark.sql(source.mkString).show(10000, false)
    }

  }
}
