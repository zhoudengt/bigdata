package com.aliyun.odps.spark.examples.ml.i2i

import org.apache.spark.sql.SparkSession

object I2I_test04 {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("I2I_test04")
      .getOrCreate()


    import spark.implicits._

    val hql1 = "select user_id,item_id  , cast(behavior_type  as DOUBLE ) as behavior_type from df_cb_125003.rec_zy_dim_user_item_behavior where length(user_id)<>0  limit 100  "
    val userDS1 = spark.sql(hql1)
    //userDS1.show(10)

    //println("  userDS1"+userDS1.show(10))

    val userDS2 = userDS1.map { row =>
      val userid = row.getString(0)
      val itemid = row.getString(1)
      val pref = row.getDouble(2)
      ItemPref(userid, itemid, pref)
    }

   // println("  userDS2"+userDS2.show(10))

    val user_ds = userDS2
    //user_ds.show(10)
    //println("  user_ds"+user_ds.show(10))
    user_ds.cache()
    user_ds.count()


    val startTime1 = System.currentTimeMillis()
    val I2I = new ItemSimilarity(spark)
    val items_similar = I2I.CooccurrenceSimilarityV1(user_ds)

    println("   --------xxxxxxxxxxxxxxxxxxxxxxxxx-----------------******************************** user_prefer_ds2"+items_similar.show(10))
    items_similar.columns

    //items_similar.cache()
    //items_similar.count
    val endTime1 = System.currentTimeMillis()
    val computeTime1 = (endTime1 - startTime1).toDouble / 1000 / 60

    //items_similar.orderBy($"similar".desc).show()

    items_similar.toDF("item_id_i","item_id_j","pref").createOrReplaceTempView("tmp_similar")

    val similar_tableName = "rec_zy_tmp_item_similar"

    println("   --------xxxxxxxxxxxxxxxxxxxxxxxxx------开始drop similar_tableName-----------**********************")
    spark.sql(s"DROP TABLE IF EXISTS ${similar_tableName}")
    println("   --------xxxxxxxxxxxxxxxxxxxxxxxxx------开始create similar_tableName -----------**********************")
    spark.sql(s"CREATE TABLE ${similar_tableName} (item_id_i STRING,item_id_j STRING, pref STRING)")
    println("   --------xxxxxxxxxxxxxxxxxxxxxxxxx------开始insert similar_tableName -----------**********************")
    spark.sql(s"insert into  TABLE ${similar_tableName}  select item_id_i ,item_id_j , pref  from tmp_similar ")


    val startTime2 = System.currentTimeMillis()
    val user_predict = I2I.Recommend(items_similar, user_ds)

    println("   --------xxxxxxxxxxxxxxxxxxxxxxxxx-----------------******************************** user_prefer_ds2"+user_predict.show(10))

    user_predict.columns
    user_predict.toDF("userid","itemid","pref").createOrReplaceTempView("tmp")
    val endTime2 = System.currentTimeMillis()
    // val computeTime2 = (endTime2 - startTime2).toDouble / 1000 / 60
    //user_predict.orderBy($"userid".asc, $"pref".desc).show(100)

    println("   --------xxxxxxxxxxxxxxxxxxxxxxxxx------开始执行-----------**********************")
    val tableName = "rec_zy_tmp_user_predict"

    println("   --------xxxxxxxxxxxxxxxxxxxxxxxxx------开始drop -----------**********************")
    spark.sql(s"DROP TABLE IF EXISTS ${tableName}")
    println("   --------xxxxxxxxxxxxxxxxxxxxxxxxx------开始create -----------**********************")
    spark.sql(s"CREATE TABLE ${tableName} (userid STRING,itemid STRING, pref STRING)")
    println("   --------xxxxxxxxxxxxxxxxxxxxxxxxx------开始insert -----------**********************")
    spark.sql(s"insert into  TABLE ${tableName}  select userid ,itemid , pref  from tmp ")

    // select *   from df_cb_125003.rec_zy_tmp_user_predict order by userid asc,pref desc

    //println("   --------xxxxxxxxxxxxxxxxxxxxxxxxx------end -----------**********************")


  }

}
