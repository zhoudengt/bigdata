package com.aliyun.odps.spark.examples.ml.i2i

import org.apache.spark.sql.SparkSession

object I2I_table {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("I2I_table")
      .getOrCreate()

    val hql1 = "select user_id,item_id  , cast(behavior_type  as DOUBLE ) as behavior_type from df_cb_125003.rec_zy_dim_user_item_behavior where length(user_id)<>0  limit 10  "
    val userDS1 = spark.sql(hql1)
    userDS1.show(10)


  }

}
