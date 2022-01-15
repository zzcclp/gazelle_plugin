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
      .config("spark.sql.files.maxPartitionBytes", 1024 << 10 << 10)  // default is 128M
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
      .config("spark.sql.parquet.columnarReaderBatchSize", "20000")
      .config("spark.plugins", "com.intel.oap.GazellePlugin")
      .config("spark.sql.execution.arrow.maxRecordsPerBatch", "20000")
      .config("spark.oap.sql.columnar.columnartorow", "false")
      .config("spark.oap.sql.columnar.use.emptyiter", "false")
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

    val spark = sessionBuilder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    // testTableScan(spark)
    // testTableScan1(spark)
    // testQ6(spark)
    testIntelQ6(spark)

    System.out.println("waiting for finishing")
    Thread.sleep(1800000)
    spark.stop()
    System.out.println("finished")
  }

  def testTableScan(spark: SparkSession): Unit = {
    val testDF = spark.read.format("arrow")
      .load("/home/myubuntu/Works/c_cpp_projects/Kyligence-ClickHouse/utils/local-engine/tests/data/iris.parquet")
    testDF.createOrReplaceTempView("chlib")
    val cnt = 1
    var minTime = Long.MaxValue
    for (i <- 1 to cnt) {
      val startTime = System.nanoTime()
      // spark.sql("select sepal_length, type, type_string from chlib").show(2, false)
      spark.sql("select * from chlib").show(1, false)
      // println(spark.sql("select * from chlib").take(10).mkString("=="))
      val tookTime = System.nanoTime() - startTime
      println(tookTime)
      if (minTime > tookTime) {
        minTime = tookTime
      }
    }
    println("min time" + minTime)

  }

  def testTableScan1(spark: SparkSession): Unit = {

    //val testDF = spark.read.format("parquet")
    //  .load("/data1/test_output/intel-gazelle-test-8m.snappy.parquet")

    val testDF = spark.read.format("arrow")
      .load("/data1/test_output/intel-gazelle-test-4m.snappy.parquet")
    testDF.printSchema()
    testDF.createOrReplaceTempView("gazelle_intel")
    val cnt = 20
    var minTime = Long.MaxValue
    for (i <- 1 to cnt) {
      val startTime = System.nanoTime()
      /* spark.sql("select l_extendedprice, l_discount, l_shipdate_new, l_returnflag from " +
        "gazelle_intel").show(200, false) */
      // select l_extendedprice, l_discount, l_shipdate_new, l_returnflag
      // select l_orderkey, l_partkey, l_suppkey, l_linenumber
      /* println(spark
        .sql(
          """
            | SELECT count(1)
            | from gazelle_intel
            | where l_orderkey is null or l_partkey is null or l_suppkey is null
            | or l_linenumber is null or l_quantity is null
            | limit 20
            |""".stripMargin).collect().mkString("==")) */
      /*println(spark
        .sql(
          """
            | SELECT l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity
            | from gazelle_intel
            | order by l_extendedprice
            | limit 20
            |""".stripMargin).collect().mkString("=="))*/
      /* println(spark
        .sql(
          """
            | SELECT l_orderkey, l_partkey, l_suppkey, l_linenumber,
            | l_quantity, l_extendedprice, l_discount, l_tax, l_shipdate_new,
            | l_commitdate_new, l_receiptdate_new
            | from gazelle_intel
            | order by l_extendedprice
            | limit 20
            |""".stripMargin).collect().mkString("==")) */
      /*println(spark
        .sql(
          """
            | SELECT l_orderkey
            | from gazelle_intel
            | order by l_orderkey
            | limit 2
            |""".stripMargin).collect().mkString("=="))*/
      println(spark
        .sql(
          """
            | SELECT *
            | from gazelle_intel
            | order by l_extendedprice
            | limit 20
            |""".stripMargin).collect().mkString("=="))
      val tookTime = System.nanoTime() - startTime
      println(tookTime)
      if (minTime > tookTime) {
        minTime = tookTime
      }
    }
    println("min time" + minTime)

  }

  def testIntelQ6(spark: SparkSession): Unit = {
    val testDF = spark.read.format("arrow")
      .load("/data1/test_output/intel-gazelle-test.snappy.parquet")
    testDF.createOrReplaceTempView("gazelle_intel")
    val cnt = 10
    val tookTimeArr = ArrayBuffer[Long]()
    for (i <- 1 to cnt) {
      val startTime = System.nanoTime()
      /*spark.sql(
        """
          | select *
          | from gazelle_intel
          |""".stripMargin).show(200, false)*/
      spark.sql(
        """
          | select sum(l_extendedprice*l_discount) as revenue
          | from gazelle_intel
          | where l_shipdate_new >= 8766 and l_shipdate_new < 9131
          | and l_discount between 0.06 - 0.01 and 0.06 + 0.01 and l_quantity < 24
          |""".stripMargin).show(200, false)
      /*spark.sql(
        """
          | select sum(l_extendedprice*l_discount) as revenue,
          | sum(l_extendedprice) as sum1,
          | sum(l_quantity) as sum2
          | from gazelle_intel
          | where l_shipdate_new >= 8766 and l_shipdate_new < 9131
          | and l_discount between 0.06 - 0.01 and 0.06 + 0.01 and l_quantity < 24
          |""".stripMargin).show(200, false)*/
      val tookTime = (System.nanoTime() - startTime) / 1000000
      tookTimeArr += tookTime
    }

    println(tookTimeArr.mkString(","))

    spark.conf.set("org.apache.spark.example.columnar.enabled", "false")
    import spark.implicits._
    val df = spark.sparkContext.parallelize(tookTimeArr.toSeq, 1).toDF("time")
    df.printSchema()
    println(df.count())
    df.summary().show(100, false)
  }

  def createTempView(spark: SparkSession, currentPath: String): Unit = {
    val parquetFilesPath = new File(currentPath + "../../../../../tpch-data")
      .getAbsolutePath
    val dataSourceMap = Map(
      "customer" -> spark.read.format("arrow").load(parquetFilesPath + "/customer"),

      "lineitem" -> spark.read.format("arrow").load(parquetFilesPath + "/lineitem"),

      "nation" -> spark.read.format("arrow").load(parquetFilesPath + "/nation"),

      "region" -> spark.read.format("arrow").load(parquetFilesPath + "/region"),

      "orders" -> spark.read.format("arrow").load(parquetFilesPath + "/order"),

      "part" -> spark.read.format("arrow").load(parquetFilesPath + "/part"),

      "partsupp" -> spark.read.format("arrow").load(parquetFilesPath + "/partsupp"),

      "supplier" -> spark.read.format("arrow").load(parquetFilesPath + "/supplier"))

    dataSourceMap.foreach {
      case (key, value) => value.createOrReplaceTempView(key)
    }
    /*spark.sql(
      """
        | show tables;
        |""".stripMargin).show(200, false)*/
  }

  def testQ6(spark: SparkSession): Unit = {
    val currentPath = this.getClass.getResource("/").getPath
    val queriesPath = currentPath + "queries/"
    createTempView(spark, currentPath)

    import scala.io.Source
    val queryId = "q%02d.sql".format(6)
    val source = Source.fromFile(new File(queriesPath + queryId), "UTF-8")
    spark.sparkContext.setJobDescription(s"Query ${queryId}")
    spark.sql(source.mkString).show(10000, false)

  }
}
