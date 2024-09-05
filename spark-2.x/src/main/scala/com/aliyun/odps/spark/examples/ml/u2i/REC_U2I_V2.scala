package com.aliyun.odps.spark.examples.ml.u2i
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator

object REC_U2I_V2 {






  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("ItemBasedCF").getOrCreate()

    // 从 Hive 表中读取用户-物品评分数据
    val ratingsDF = spark.sql("SELECT user_id, item_id, rating FROM ratings_table")

    // 案例数据假设如下：
    // | user_id | item_id | rating |
    // | 1 | 101 | 4 |
    // | 1 | 102 | 3 |
    // | 2 | 101 | 5 |
    // | 2 | 103 | 4 |

    // 使用 ALS 进行训练，这里主要是为了得到物品之间的潜在关系
    // 使用 ALS（Alternating Least Squares）算法进行训练，主要是为了得到物品之间的潜在关系。
    val als = new ALS()
      .setMaxIter(10)
      .setRegParam(0.1)
      .setUserCol("user_id")
      .setItemCol("item_id")
      .setRatingCol("rating")

    val model = als.fit(ratingsDF)

    // 得到物品之间的相似度矩阵 从训练好的模型中获取物品的特征向量，计算物品之间的相似度矩阵。
    val itemFactors = model.itemFactors
    val itemSimilarities = itemFactors.join(itemFactors, "id")
      .selectExpr("id as item1", "id2 as item2", "dot_product(vector1, vector2) as similarity")

    // 为用户进行推荐 为每个用户根据物品相似度进行推荐。
    val usersDF = ratingsDF.select("user_id").distinct()
    val recommendationsDF = usersDF.join(itemSimilarities, usersDF("user_id") === itemSimilarities("item1"))
      .select("user_id", "item2", "similarity")
      .orderBy(desc("similarity"))


    // 将推荐结果写回 Hive 表
    recommendationsDF.write.mode("overwrite").saveAsTable("recommended_items_table")

    // 评估算法

    // 使用回归评估器评估算法，计算均方根误差（RMSE）来衡量预测值与真实值之间的差距。
    // 评估部分可以根据实际情况选择不同的评估指标，如准确率、召回率等，以全面评估算法的性能。同时，还可以通过调整 ALS 算法的参数、增加数据预处理步骤等方式来优化算法的效果。

    val predictions = model.transform(ratingsDF)
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error = $rmse")
  }
}




