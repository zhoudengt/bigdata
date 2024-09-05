package com.aliyun.odps.spark.examples.ml.u2i

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions.{col, explode}

/**
 * 基于 Scala 的用户协同过滤算法示例
 */
object REC_U2I {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("REC_U2I").getOrCreate()

    import spark.implicits._

    // 从 Hive 表中读取数据
    val data = spark.sql("select user_id ,item_id  , cast(behavior_type  as DOUBLE ) as rating from df_cb_125003.rec_zy_dim_user_item_behavior where length(user_id)<>0    limit 1000 ")

    /**
     * 对用户 ID 进行字符串索引编码
     * 将字符串形式的用户 ID 转换为数值索引，以便模型处理
     */

    val indexerUser = new StringIndexer()
      .setInputCol("user_id")
      .setOutputCol("userIndex")

    val indexerItem = new StringIndexer()
      .setInputCol("item_id")
      .setOutputCol("itemIndex")

    /**
     * 应用用户 ID 索引编码
     * 对原始数据进行转换，得到包含用户索引的新数据
     */
    val indexedData = indexerUser.fit(data).transform(data)
    val finalData = indexerItem.fit(indexedData).transform(indexedData)

    // 将数据分为训练集和测试集，比例分别为 80% 和 20%
    val Array(training, test) = finalData.randomSplit(Array(0.8, 0.2))

    /**
     * 创建 ALS 模型
     * 设置最大迭代次数为 5 次
     * 设置正则化参数为 0.01
     * 指定用户索引列、商品索引列和评分列
     */

    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userIndex")
      .setItemCol("itemIndex")
      .setRatingCol("rating")

    /**
     * 训练 ALS 模型
     * 使用训练集数据对模型进行训练     // 使用 ALS（Alternating Least Squares）算法进行训练，这是一种实现 LFM 的常见算法

     */
    val model = als.fit(training)

    // 为所有用户生成推荐
    val allUsers = finalData.select("userIndex").distinct()

    val recommendationsDF = model.recommendForAllUsers(10)  // 为每个用户推荐 10 个物品


    // 使用 explode 函数将 recommendations 列展开
    val explodedRecommendationsDF = recommendationsDF.select(col("userIndex"), explode(col("recommendations")).as("explodedRecommendation"))

    // 进一步提取展开后的列中的 itemIndex 和 rating
    val finalDF = explodedRecommendationsDF.select(col("userIndex"), col("explodedRecommendation.itemIndex").as("itemIndex"), col("explodedRecommendation.rating").as("rating"))

    finalDF.show()
    //finalDF.write.mode("overwrite").saveAsTable("your_hive_table_name")
    finalData.show()
    //finalData.write.mode("overwrite").saveAsTable("your_hive_dim_name")

    spark.stop()

  }
}