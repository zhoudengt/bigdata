package com.aliyun.odps.spark.examples.ml.tf


import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, StringType}

import java.util
import org.apache.spark.sql.Row

import scala.collection.JavaConversions.asScalaBuffer


// 导入 jieba 的相关类
import com.huaban.analysis.jieba.JiebaSegmenter

// 可以根据聚类结果为用户提供个性化的商品推荐。例如，如果一个用户购买了某个被标记为 “1” 聚类的商品，那么可以向该用户推荐同一聚类中的其他商品。因为这些商品具有相似的特征，用户可能对它们也感兴趣。
// 基于内容的推荐  需要在pom中导入 com.huaban.analysis.jieba.JiebaSegmenter 类
object tf_idf {
  def main(args: Array[String]): Unit = {
    // 创建 SparkSession 对象
    val spark = SparkSession
      .builder()
      .appName("tf_idf")
      .getOrCreate()


    // 从 Hive 表中读取商品数据
    val productsDF = spark.sql("SELECT product_id, trim(product_name) as product_name  FROM df_cb_125003.rec_zy_tmp_tf_idf1 ")

    // 定义一个不可拆分词组的集合
    val unbreakableWords = Set("四合一", "爱车帮")

    // 定义分词函数，避免拆分特定词组
    def jiebaTokenize(text: String): Seq[String] = {
      // 预处理文本，将不可拆分词组替换为特殊标记
      val preprocessedText = unbreakableWords.foldLeft(text) { (acc, word) =>
        acc.replace(word, s"[${word}]")
      }
      val segmenter = new JiebaSegmenter()
      val words = segmenter.sentenceProcess(preprocessedText)
      words.toSeq.map { word =>
        if (word.startsWith("[") && word.endsWith("]")) {
          // 如果是特殊标记，提取出原始的不可拆分词组
          word.substring(1, word.length - 1)
        } else {
          word
        }
      }
    }



    // 将 jieba 分词函数注册为 UDF
    val jiebaTokenizeUDF = udf(jiebaTokenize _)

    // 对商品名称进行分词
    val productsWithTokensDF = productsDF.withColumn("tokens", jiebaTokenizeUDF(col("product_name")))

    //productsWithTokensDF.show(10,false)

    // 计算词频
    val hashingTF = new HashingTF().setInputCol("tokens").setOutputCol("rawFeatures")
    val featurizedData = hashingTF.transform(productsWithTokensDF)

    //featurizedData.show(100,false)

    // 计算逆文档频率
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)

    rescaledData.show(10,false)

    // 使用 K-Means 进行聚类
    val kmeans = new KMeans().setK(10).setSeed(3L)
    val model = kmeans.fit(rescaledData.select("features"))

    // 为每个商品分配聚类标签
    val clusteredData = model.transform(rescaledData).withColumnRenamed("prediction", "cluster")

    clusteredData.show(1000,false)
    // 根据聚类结果生成推荐
    //val recommendedProductsDF = clusteredData.groupBy("cluster").agg(collect_list("product_id").as("recommended_products"))
    //recommendedProductsDF.show(10,false)

    // 将推荐结果写入 Hive 表
    //recommendedProductsDF.write.mode("overwrite").saveAsTable("recommended_products_table")
    spark.stop()

  }
}