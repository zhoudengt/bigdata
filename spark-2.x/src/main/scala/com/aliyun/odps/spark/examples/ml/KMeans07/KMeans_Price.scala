package com.aliyun.odps.spark.examples.ml.KMeans07



import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler

object KMeans_Price {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("KMeans_Price").getOrCreate()

    // 从 Hive 表读取数据
    val inputData = spark.sql("SELECT goods_id, price FROM df_cb_125003.rec_zy_tmp_kmeans_price  ")

    // 创建特征向量
    val assembler = new VectorAssembler()
      .setInputCols(Array("price"))
      .setOutputCol("features")

    val assembledData = assembler.transform(inputData)

    assembledData.show(10,false)

    // 进行 K-Means 聚类
    val kmeans = new KMeans().setK(10).setSeed(3L).setFeaturesCol("features")
    val model = kmeans.fit(assembledData)

    // 添加聚类结果列
    //val clusteredData = model.transform(assembledData).withColumnRenamed("prediction", "cluster")

    //clusteredData.show(1000,false)



    // 将向量类型的列转换为数组类型
    val clusteredData = model.transform(assembledData)
      .withColumn("features_array", expr("features.toArray"))
      .drop("features")
      .withColumnRenamed("prediction", "cluster")

    // 将结果写入 Hive 表
    //clusteredData.write.mode("overwrite").saveAsTable("rec_zy_tmp_kmeans_price_result")

    spark.stop()
  }
}