package com.intel.oap.tpch

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

import java.sql.Date

import scala.io.Source

import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions.col

// TPC-H table schemas
case class Customer(
                     c_custkey: Long,
                     c_name: String,
                     c_address: String,
                     c_nationkey: Long,
                     c_phone: String,
                     c_acctbal: Double,
                     c_mktsegment: String,
                     c_comment: String)

case class Lineitem(
                     l_orderkey: Long,
                     l_partkey: Long,
                     l_suppkey: Long,
                     l_linenumber: Long,
                     l_quantity: Double,
                     l_extendedprice: Double,
                     l_discount: Double,
                     l_tax: Double,
                     l_returnflag: String,
                     l_linestatus: String,
                     l_shipdate: Date,
                     l_commitdate: Date,
                     l_receiptdate: Date,
                     l_shipinstruct: String,
                     l_shipmode: String,
                     l_comment: String)

case class Nation(
                   n_nationkey: Long,
                   n_name: String,
                   n_regionkey: Long,
                   n_comment: String)

case class Order(
                  o_orderkey: Long,
                  o_custkey: Long,
                  o_orderstatus: String,
                  o_totalprice: Double,
                  o_orderdate: Date,
                  o_orderpriority: String,
                  o_clerk: String,
                  o_shippriority: Long,
                  o_comment: String)

case class Part(
                 p_partkey: Long,
                 p_name: String,
                 p_mfgr: String,
                 p_brand: String,
                 p_type: String,
                 p_size: Long,
                 p_container: String,
                 p_retailprice: Double,
                 p_comment: String)

case class Partsupp(
                     ps_partkey: Long,
                     ps_suppkey: Long,
                     ps_availqty: Long,
                     ps_supplycost: Double,
                     ps_comment: String)

case class Region(
                   r_regionkey: Long,
                   r_name: String,
                   r_comment: String)

case class Supplier(
                     s_suppkey: Long,
                     s_name: String,
                     s_address: String,
                     s_nationkey: Long,
                     s_phone: String,
                     s_acctbal: Double,
                     s_comment: String)

case class TableConfig(repartitionNum: Int, sortedCols: Seq[String])

object ImportCsvToParquet {

  def main(args: Array[String]): Unit = {

    val (csvFilesPath, parquetFilesPath,
      repartitionNum, sortOrNot,
      configed, configFile) = if (args.length > 0) {
      (args(0), args(1), args(2).toInt, args(3).toBoolean, true, args(4))
    } else {
      ("/data1/tpch-data-gen/tpch_2_16_0/sf10",
        "/data1/test_output/tpch-data-sf10-morepage",
        2, false, false, this.getClass.getResource("/").getPath + "/import_csv_to_parquet.config")
    }

    var tableConfigs: Map[String, TableConfig] = Map()
    for (line <- Source.fromFile(configFile).getLines()) {
      val lineArr = line.split("=")
      val tableKey = lineArr(0)
      val tableConfigArr = lineArr(1).split(";")
      val tableConfig = if (tableConfigArr.length > 1) {
        TableConfig(tableConfigArr(0).toInt, tableConfigArr(1).split(",").toSeq)
      } else {
        TableConfig(tableConfigArr(0).toInt, Seq.empty[String])
      }
      tableConfigs += (tableKey -> tableConfig)
    }

    val sessionBuilderTmp = SparkSession
      .builder()
      .appName("Import-TPCH")
    val sessionBuilder = if (!configed) {
      sessionBuilderTmp
        .master("local[3]")
        .config("spark.driver.memory", "8G")
        .config("spark.driver.memoryOverhead", "4G")
        .config("spark.default.parallelism", 3)
        .config("spark.sql.shuffle.partitions", 3)
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.files.maxPartitionBytes", 256 << 10 << 10) // default is 128M
        .config("spark.sql.files.minPartitionNum", "1")
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.sql.parquet.enableVectorizedReader", "true")
        // 536870912
        //.config("spark.hadoop.parquet.block.size", "8388608")
        //.config("spark.hadoop.parquet.page.size", "1048576")
        //.config("spark.hadoop.parquet.page.size", "131072")
        //.config("spark.hadoop.parquet.dictionary.page.size", "1048576")
        //.config("spark.hadoop.parquet.writer.version",
        //  ParquetProperties.WriterVersion.PARQUET_2_0.toString)
        //.config("spark.sql.parquet.compression.codec", "none")
        .config("spark.memory.fraction", "0.3")
        .config("spark.memory.storageFraction", "0.3")
        .config("spark.sql.planChangeLog.level", "info")
    } else {
      sessionBuilderTmp
    }

    val spark = sessionBuilder.getOrCreate()
    if (!configed) {
      spark.sparkContext.setLogLevel("WARN")
    }

    importCsvToParquet(spark, csvFilesPath, parquetFilesPath,
      repartitionNum, sortOrNot, tableConfigs)

    System.out.println("waiting for finishing")
    Thread.sleep(1800000)
    spark.stop()
    System.out.println("finished")
  }

  def importCsvToParquet(spark: SparkSession,
                         csvFilesPath: String, parquetFilesPath: String,
                         repartitionNum: Int, sortOrNot: Boolean,
                         tableConfigs: Map[String, TableConfig]): Unit = {
    val format = new java.text.SimpleDateFormat("yyyy-MM-dd")

    import spark.implicits._
    val dfMap = Map(
      "customer" -> spark.sparkContext.textFile(csvFilesPath + "/customer.tbl").map(_.split('|'))
        .map(p =>
        Customer(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim.toLong, p(4).trim, p(5).trim.toDouble, p(6).trim, p(7).trim)).toDF(),

      "lineitem" -> spark.sparkContext.textFile(csvFilesPath + "/lineitem.tbl").map(_.split('|'))
        .map(p =>
        Lineitem(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toLong,
          p(4).trim.toDouble, p(5).trim.toDouble, p(6).trim.toDouble, p(7).trim.toDouble,
          p(8).trim, p(9).trim,
          new Date(format.parse(p(10).trim).getTime),
          new Date(format.parse(p(11).trim).getTime),
          new Date(format.parse(p(12).trim).getTime), p(13).trim, p(14).trim, p(15).trim)).toDF(),

      "nation" -> spark.sparkContext.textFile(csvFilesPath + "/nation.tbl").map(_.split('|'))
        .map(p =>
        Nation(p(0).trim.toLong, p(1).trim, p(2).trim.toLong, p(3).trim)).toDF(),

      "region" -> spark.sparkContext.textFile(csvFilesPath + "/region.tbl").map(_.split('|'))
        .map(p =>
        Region(p(0).trim.toLong, p(1).trim, p(2).trim)).toDF(),

      "order" -> spark.sparkContext.textFile(csvFilesPath + "/orders.tbl").map(_.split('|'))
        .map(p =>
        Order(p(0).trim.toLong, p(1).trim.toLong, p(2).trim, p(3).trim.toDouble, new Date(format.parse(p(4).trim).getTime),
          p(5).trim, p(6).trim, p(7).trim.toLong, p(8).trim)).toDF(),

      "part" -> spark.sparkContext.textFile(csvFilesPath + "/part.tbl").map(_.split('|')).map(p =>
        Part(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim, p(5).trim.toLong, p(6).trim, p(7).trim.toDouble, p(8).trim)).toDF(),

      "partsupp" -> spark.sparkContext.textFile(csvFilesPath + "/partsupp.tbl").map(_.split('|'))
        .map(p =>
        Partsupp(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toDouble, p(4).trim)).toDF(),

      "supplier" -> spark.sparkContext.textFile(csvFilesPath + "/supplier.tbl").map(_.split('|'))
        .map(p =>
        Supplier(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim.toLong, p(4).trim, p(5).trim.toDouble, p(6).trim)).toDF())

    dfMap.foreach {
      case (key, value) => {
        value.printSchema()
        val repartitionDF = if (sortOrNot) {
          val tableConfig = tableConfigs(key)
          if (tableConfig.sortedCols.isEmpty) {
            value.repartition(tableConfigs(key).repartitionNum)
          } else if (tableConfig.sortedCols.length == 1) {
            value.repartition(tableConfigs(key).repartitionNum)
              .sortWithinPartitions(tableConfig.sortedCols(0))
          } else {
            value.repartition(tableConfigs(key).repartitionNum)
              .sortWithinPartitions(tableConfig.sortedCols.map(col(_)) : _*)
          }
        } else {
          value.repartition(tableConfigs(key).repartitionNum)
        }

        repartitionDF.write.mode(SaveMode.Overwrite)
            .parquet(parquetFilesPath + "/" + key + "/")
      }
    }

    dfMap.foreach {
      case (key, value) => value.createOrReplaceTempView(key)
    }

    spark.sql(
      """
        | show tables;
        |""".stripMargin).show(200, false)

    dfMap.foreach {
      case (key, value) => {
        spark.sql(
          s"""
            | select count(1) from ${key}
            |""".stripMargin).show(10, false)
        spark.sql(
          s"""
             | select * from ${key}
             |""".stripMargin).show(10, false)
      }
    }

    spark.sql(
      s"""
         | select min(l_shipdate), max(l_shipdate),min(l_receiptdate), max(l_receiptdate) from lineitem
         |""".stripMargin).show(10, false)
    spark.sql(
      s"""
         | select min(o_orderdate), max(o_orderdate) from order
         |""".stripMargin).show(10, false)
  }
}
