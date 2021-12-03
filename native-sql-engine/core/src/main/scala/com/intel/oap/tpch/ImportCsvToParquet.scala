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

import org.apache.spark.sql.{SaveMode, SparkSession}

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

object ImportCsvToParquet {

  def main(args: Array[String]): Unit = {
    val sessionBuilder = SparkSession
      .builder()
      .master("local[1]")
      .appName("Import-TPCH")
      .config("spark.driver.memory", "8G")
      .config("spark.driver.memoryOverhead", "2G")
      .config("spark.default.parallelism", 1)
      .config("spark.sql.shuffle.partitions", 1)
      .config("spark.sql.adaptive.enabled", "false")
      .config("spark.sql.files.maxPartitionBytes", 256 << 10 << 10)  // default is 128M
      .config("spark.sql.files.minPartitionNum", "1")
      .config("spark.sql.parquet.filterPushdown", "true")
      .config("spark.sql.parquet.enableVectorizedReader", "true")
      //.config("spark.hadoop.parquet.block.size", "8388608")
      .config("spark.memory.fraction", "0.3")
      .config("spark.memory.storageFraction", "0.3")
      .config("spark.sql.planChangeLog.level", "info")

    val spark = sessionBuilder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    importCsvToParquet(spark)

    System.out.println("waiting for finishing")
    Thread.sleep(1800000)
    spark.stop()
    System.out.println("finished")
  }

  def importCsvToParquet(spark: SparkSession): Unit = {
    val csvFilesPath = "/data1/tpch-data-gen/tpch_2_16_0/sf0-1"
    val parquetFilesPath = "/home/myubuntu/Works/IdeaProjects/intel-native-sql-engine/tpch-data"
    val format = new java.text.SimpleDateFormat("yyyy-MM-dd")

    import spark.implicits._
    val dfMap = Map(
      "customer" -> spark.sparkContext.textFile(csvFilesPath + "/customer.csv").map(_.split('|'))
        .map(p =>
        Customer(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim.toLong, p(4).trim, p(5).trim.toDouble, p(6).trim, p(7).trim)).toDF(),

      "lineitem" -> spark.sparkContext.textFile(csvFilesPath + "/lineitem.csv").map(_.split('|'))
        .map(p =>
        Lineitem(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toLong,
          p(4).trim.toDouble, p(5).trim.toDouble, p(6).trim.toDouble, p(7).trim.toDouble,
          p(8).trim, p(9).trim,
          new Date(format.parse(p(10).trim).getTime),
          new Date(format.parse(p(11).trim).getTime),
          new Date(format.parse(p(12).trim).getTime), p(13).trim, p(14).trim, p(15).trim)).toDF(),

      "nation" -> spark.sparkContext.textFile(csvFilesPath + "/nation.csv").map(_.split('|'))
        .map(p =>
        Nation(p(0).trim.toLong, p(1).trim, p(2).trim.toLong, p(3).trim)).toDF(),

      "region" -> spark.sparkContext.textFile(csvFilesPath + "/region.csv").map(_.split('|'))
        .map(p =>
        Region(p(0).trim.toLong, p(1).trim, p(2).trim)).toDF(),

      "order" -> spark.sparkContext.textFile(csvFilesPath + "/orders.csv").map(_.split('|'))
        .map(p =>
        Order(p(0).trim.toLong, p(1).trim.toLong, p(2).trim, p(3).trim.toDouble, new Date(format.parse(p(4).trim).getTime),
          p(5).trim, p(6).trim, p(7).trim.toLong, p(8).trim)).toDF(),

      "part" -> spark.sparkContext.textFile(csvFilesPath + "/part.csv").map(_.split('|')).map(p =>
        Part(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim, p(5).trim.toLong, p(6).trim, p(7).trim.toDouble, p(8).trim)).toDF(),

      "partsupp" -> spark.sparkContext.textFile(csvFilesPath + "/partsupp.csv").map(_.split('|'))
        .map(p =>
        Partsupp(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toDouble, p(4).trim)).toDF(),

      "supplier" -> spark.sparkContext.textFile(csvFilesPath + "/supplier.csv").map(_.split('|'))
        .map(p =>
        Supplier(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim.toLong, p(4).trim, p(5).trim.toDouble, p(6).trim)).toDF())

    dfMap.foreach {
      case (key, value) => {
        value.printSchema()
        value.repartition(1).write.mode(SaveMode.Overwrite)
          .parquet(parquetFilesPath + "/" + key + "/")
      }
    }

    /* dfMap.foreach {
      case (key, value) => value.createOrReplaceTempView(key)
    }

    spark.sql(
      """
        | show tables;
        |""".stripMargin).show(200, false)
    spark.sql(
      """
        | select count(1) from lineitem
        |""".stripMargin).show(200, false)*/
  }
}
