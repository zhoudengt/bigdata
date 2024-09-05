package com.aliyun.odps.spark.examples.ml.logistic05

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object GBDT_LR {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("GBDT_LR")
      .getOrCreate()

    // 从 Hive 表中读取数据
    val data = spark.sql("SELECT feature1, feature2, feature3, label FROM your_table_name")

    // 定义特征列名
    val featureCols = Array("feature1", "feature2", "feature3")

    // 将特征列组合成一个向量列
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val assembledData = assembler.transform(data)

    // 划分训练集和测试集
    val Array(trainingData, testData) = assembledData.randomSplit(Array(0.7, 0.3))

    // 训练 GBDT 模型
    //val gbdt = new GBDTClassifier().setLabelCol("label").setFeaturesCol("features")
    //val gbdtModel = gbdt.fit(trainingData)

    // 提取 GBDT 模型的特征
    //val gbdtFeatures = gbdtModel.transform(trainingData).select("label", "prediction", gbdtModel.featureImportances)

    // 将 GBDT 模型的特征与原始特征组合
    val lrData = assembler.transform(trainingData)

    // 训练 LR 模型
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features")
    val lrModel = lr.fit(lrData)

    // 使用测试集进行预测
    val predictions = lrModel.transform(testData)

    // 将预测结果写入 Hive 表
    predictions.write.mode("overwrite").saveAsTable("your_result_table_name")

    // 关闭 SparkSession
    spark.stop()

  }
}
