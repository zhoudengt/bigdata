package com.aliyun.odps.spark.examples.ml.i2i
/**
 * 用户评分.
 * @param userid 用户
 * @param itemid 评分物品
 * @param pref 评分
 */
case class ItemPref(
                     val userid: String,
                     val itemid: String,
                     val pref: Double)

/**
 * 相似度.
 * @param itemidI 物品
 * @param itemidJ 物品
 * @param similar 相似度
 */
case class ItemSimi(
                     val itemidI: String,
                     val itemidJ: String,
                     val similar: Double)

/**
 * 推荐结果.
 * @param userid 用户
 * @param itemid 物品
 * @param pref 得分
 */
case class UserRecomm(
                       val userid: String,
                       val itemid: String,
                       val pref: Double)
class ItemSimilarity(@transient val myspark: org.apache.spark.sql.SparkSession) extends Serializable {

  import myspark.implicits._
  import org.apache.spark.sql.functions._
  /**
   * 同现相似度矩阵计算.
   * w(i,j) = N(i)∩N(j)/sqrt(N(i)*N(j))

   *
   */
  def CooccurrenceSimilarityV1(user_ds: org.apache.spark.sql.Dataset[ItemPref]): org.apache.spark.sql.Dataset[ItemSimi] = {


    val user_ds_i = user_ds.withColumnRenamed("itemid", "itemidI").withColumnRenamed("pref", "prefI")
    val user_ds_j = user_ds.withColumnRenamed("itemid", "itemidJ").withColumnRenamed("pref", "prefJ")


/**
    user_ds_i.toDF("userid","itemidI","prefI").createOrReplaceTempView("tmp1")
    val tableNamei = "rec_zy_tmp_user_ds_i"
    myspark.sql(s"DROP TABLE IF EXISTS ${tableNamei}")
    myspark.sql(s"CREATE TABLE ${tableNamei} (userid STRING,itemidI STRING, prefI double)")
    myspark.sql(s"insert into  TABLE ${tableNamei}  select userid ,itemidI , prefI  from tmp1 ")

    user_ds_j.toDF("userid","itemidI","prefI").createOrReplaceTempView("tmp1")
    val tableNamej = "rec_zy_tmp_user_ds_j"
    myspark.sql(s"DROP TABLE IF EXISTS ${tableNamej}")
    myspark.sql(s"CREATE TABLE ${tableNamej} (userid STRING,itemidJ STRING, prefJ double)")
    myspark.sql(s"insert into  TABLE ${tableNamej}  select userid ,itemidJ , prefJ  from tmp1 ")
*/



    // 1 (用户：物品) 笛卡尔积 (用户：物品) => 物品:物品组合
    val user_ds1 = user_ds_i.join(user_ds_j, "userid")

    //withColumn 对数据集增加列
    val user_ds2 = user_ds1.withColumn("score", col("prefI") * 0 + 1).select("itemidI", "itemidJ", "score")




    // 2 物品:物品:频次
    val user_ds3 = user_ds2.groupBy("itemidI", "itemidJ").agg(sum("score").as("sumIJ"))
    //println(" 2 物品:物品:频次 user_ds3"+user_ds3.show(10))


    // 3 对角矩阵
    val user_ds4 = user_ds3.where("itemidI = itemidJ")

    //println(" 3 对角矩阵 user_ds4"+user_ds4.show(10))


    // 4 非对角矩阵
    val user_ds5 = user_ds3.filter("itemidI != itemidJ")
   // println(" 4 非对角矩阵 user_ds5"+user_ds5.show(10))


    // 5 计算同现相似度（物品1，物品2，同现频次）
    val user_ds6 = user_ds5.join(user_ds4.withColumnRenamed("sumIJ", "sumJ").select("itemidJ", "sumJ"), "itemidJ")

   // println(" 5 计算同现相似度 user_ds6"+user_ds6.show(10))

    val user_ds7 = user_ds6.join(user_ds4.withColumnRenamed("sumIJ", "sumI").select("itemidI", "sumI"), "itemidI")
   // println(" 5 计算同现相似度 user_ds7"+user_ds7.show(10))
    val user_ds8 = user_ds7.withColumn("result", col("sumIJ") / sqrt(col("sumI") * col("sumJ")))

   // println(" 5 计算同现相似度（物品1，物品2，同现频次） user_ds8"+user_ds8.show(10))

    // 6 结果返回
    val out = user_ds8.select("itemidI", "itemidJ", "result").map { row =>
      val itemidI = row.getString(0)
      val itemidJ = row.getString(1)
      val similar = row.getDouble(2)
      ItemSimi(itemidI, itemidJ, similar)
    }
    out
  }


  /**
   * 计算推荐结果.
   * 分子  物品ihe物品j共同出现的次数
   * 分母  物品i的出现次数与物品j的出现次数乘积 开根号
   * w(i,j) = N(i)∩N(j)/sqrt(N(i)*N(j))
   * @param items_similar 物品相似矩阵
   * @param user_prefer 用户评分表
   *
   */
  def Recommend(items_similar: org.apache.spark.sql.Dataset[ItemSimi],
                user_prefer: org.apache.spark.sql.Dataset[ItemPref]): org.apache.spark.sql.Dataset[UserRecomm] = {
    //   0 数据准备
    val items_similar_ds1 = items_similar
    val user_prefer_ds1 = user_prefer



    //   1 根据用户的item召回相似物品
    val user_prefer_ds2 = items_similar_ds1.join(user_prefer_ds1, $"itemidI" === $"itemid", "inner")

    //   2 计算召回的用户物品得分

    val user_prefer_ds3 = user_prefer_ds2.withColumn("score", col("pref") * col("similar")).select("userid", "itemidJ", "score")
    //    user_prefer_ds3.show()
    //   3 得分汇总


    val user_prefer_ds4 = user_prefer_ds3.groupBy("userid", "itemidJ").agg(sum("score").as("score")).withColumnRenamed("itemidJ", "itemid")
    //    user_prefer_ds4.show()
    //   4 用户得分排序结果


    val user_prefer_ds5 = user_prefer_ds4

    //  5 结果返回

    val out1 = user_prefer_ds5.select("userid", "itemid", "score").map { row =>
      val userid = row.getString(0)
      val itemid = row.getString(1)
      val pref = row.getDouble(2)
      UserRecomm(userid, itemid, pref)
    }
    out1
  }

}



