package com.aliyun.odps.spark.examples.ml.u2i

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{ArrayType, StringType, StructType}

//电商行业的 FP-Growth算法，数据源来源于hive表，结果写入hive
//购物车频繁模式 用于套餐推荐

object FP_Growth {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("FP_Growth")
      .getOrCreate()

    import spark.implicits._

// create table df_cb_125003.rec_zy_tmp_fp_growth  as  select user_id as tran_id,wm_concat(distinct ',', goods_name) as items   from df_cb_125003.dwd_trd_mall_order_dd where dt='20240822' group by user_id  limit 1000
// 从 Hive 表中读取数据，假设表名为"your_table_name"，表中有"transaction_id"和"item"两列
    val data = spark.sql("select tran_id, items from df_cb_125003.rec_zy_tmp_fp_growth  where items  is not null  ")

    // 将数据转换为 FP-Growth 算法所需的格式，即 RDD[List[String]]，其中每个列表代表一个事务
    val transactionsRDD = data.rdd.map(row => row.getAs[String](1).split(",").toList)


    // 定义 Dataset 的 schema
    val schema = new StructType().add("items", ArrayType(StringType))


    // 将 RDD 转换为 Dataset
    val transactions = spark.createDataFrame(transactionsRDD.map(Row(_)), schema)

    //transactions.show()


    // 训练 FP-Growth 模型，设置最小支持度为 0.3，最小置信度为 0.6
    // 不满足条件会在这一块过滤数据
    // 合理设计购物车
    val fpgrowth = new FPGrowth().setMinSupport(0.1).setMinConfidence(0.1)
    val model = fpgrowth.fit(transactions)

    // 显示频繁项集
    val frequentItemsets = model.freqItemsets

    println("显示频繁项集"+frequentItemsets.show(false))



    // 显示关联规则
    val associationRules = model.associationRules
    associationRules.show()
    // 将结果转换为 DataFrame，以便后续写入 Hive 表
    import spark.implicits._
    val resultDF = frequentItemsets.toDF("items", "support")

    // 将结果写入 Hive 表，假设表名为"your_result_table_name"，写入模式为覆盖写入
    //resultDF.write.mode("overwrite").saveAsTable("your_result_table_name")

    //resultDF.show()

    // 关闭 SparkSession，释放资源
    spark.stop()
  }
}






