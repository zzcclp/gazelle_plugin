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

object ChAsLibDemo {

  def main(args: Array[String]): Unit = {
    val warehouse = "/data1/spark31_data/spark-warehouse"
    val metaStorePathAbsolute = new File("/data1/spark31_data/meta").getCanonicalPath
    val hiveMetaStoreDB = metaStorePathAbsolute + "/metastore_db"

    val sessionBuilder = SparkSession
      .builder()
      .master("local[1]")
      .appName("CH-As-Lib-Demo")
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
      .config("spark.sql.files.maxPartitionBytes", 256 << 10 << 10)  // default is 128M
      .config("spark.sql.files.minPartitionNum", "1")
      .config("spark.sql.parquet.filterPushdown", "true")
      //.config("spark.locality.wait", "0s")
      //.config("spark.sql.sources.ignoreDataLocality", "false")
      .config("spark.sql.parquet.enableVectorizedReader", "true")
      //.config("spark.hadoop.parquet.block.size", "8388608")
      //.config("spark.sql.crossJoin.enabled", "true")
      //.config("spark.sql.autoBroadcastJoinThreshold", "-1")
      .config("spark.sql.sources.useV1SourceList", "avro")
      .config("spark.memory.fraction", "0.3")
      .config("spark.memory.storageFraction", "0.3")
      .config("spark.sql.parquet.columnarReaderBatchSize", "20000")
      .config("spark.plugins", "com.intel.oap.GazellePlugin")
      .config("spark.sql.execution.arrow.maxRecordsPerBatch", "20000")
      .config("spark.oap.sql.columnar.columnartorow", "false")
      .config("spark.sql.planChangeLog.level", "info")
      .config("spark.memory.offHeap.enabled", "true")
      .config("spark.memory.offHeap.size", "6442450944")
      //.config("javax.jdo.option.ConnectionURL", s"jdbc:derby:;databaseName=$hiveMetaStoreDB;" +
      //  s"create=true")
      //.enableHiveSupport()

    val spark = sessionBuilder.getOrCreate()
    spark.sparkContext.setLogLevel("INFO")

    testTableScan(spark)
    testTableScan1(spark)

    System.out.println("waiting for finishing")
    Thread.sleep(1800000)
    spark.stop()
    System.out.println("finished")
  }

  def testTableScan(spark: SparkSession): Unit = {
    val testDF = spark.read.format("arrow")
      .load("/home/myubuntu/Works/c_cpp_projects/Kyligence-ClickHouse/utils/local-engine/tests/data/iris.parquet")
    testDF.createOrReplaceTempView("chlib")
    val cnt = 5
    var minTime = Long.MaxValue
    for (i <- 1 to cnt) {
      val startTime = System.nanoTime()
      spark.sql("select sepal_length, type from chlib").show(200, false)
      val tookTime = System.nanoTime() - startTime
      println(tookTime)
      if (minTime > tookTime) {
        minTime = tookTime
      }
    }
    println("min time" + minTime)

  }

  def testTableScan1(spark: SparkSession): Unit = {
    val testDF = spark.read.format("arrow").load("/data1/test_output/intel-gazelle-test.snappy.parquet")
    testDF.printSchema()
    testDF.createOrReplaceTempView("gazelle_intel")
    val cnt = 5
    var minTime = Long.MaxValue
    for (i <- 1 to cnt) {
      val startTime = System.nanoTime()
      spark.sql("select l_partkey, l_quantity, l_returnflag, l_receiptdate_new, l_shipinstruct from " +
        "gazelle_intel").show(200, false)
      val tookTime = System.nanoTime() - startTime
      println(tookTime)
      if (minTime > tookTime) {
        minTime = tookTime
      }
    }
    println("min time" + minTime)

  }
}
