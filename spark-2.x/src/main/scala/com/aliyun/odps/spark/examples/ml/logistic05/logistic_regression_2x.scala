package com.aliyun.odps.spark.examples.ml.logistic05

import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{ BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession

object logistic_regression_2x {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().
      master("local").
      appName("my App Name").
      getOrCreate()

    import spark.implicits._

    //1 训练样本准备
    val training = spark.read.format("libsvm").load("hdfs://1.1.1.1:9000/user/sample_libsvm_data.txt")
    training.show

    //2 建立逻辑回归模型
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    //2 根据训练样本进行模型训练
    val lrModel = lr.fit(training)

    //2 打印模型信息
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    println(s"Intercept: ${lrModel.intercept}")

    //3 建立多元回归模型
    val mlr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")

    //3 根据训练样本进行模型训练
    val mlrModel = mlr.fit(training)

    //3 打印模型信息
    println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
    println(s"Multinomial intercepts: ${mlrModel.interceptVector}")

    //4 测试样本
    val test = spark.createDataFrame(Seq(
      (1.0, Vectors.sparse(692, Array(10, 20, 30), Array(-1.0, 1.5, 1.3))),
      (0.0, Vectors.sparse(692, Array(45, 175, 500), Array(-1.0, 1.5, 1.3))),
      (1.0, Vectors.sparse(692, Array(100, 200, 300), Array(-1.0, 1.5, 1.3))))).toDF("label", "features")
    test.show

    //5 对模型进行测试
    val test_predict = lrModel.transform(test)
    test_predict.show
    test_predict.select("features", "label", "probability", "prediction").collect().foreach {
      case Row(features: Vector, label: Double, prob: Vector, prediction: Double) =>
        println(s"($features, $label) -> prob=$prob, prediction=$prediction")
    }

    //6 模型摘要
    val trainingSummary = lrModel.summary

    //6 每次迭代目标值
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))

    //6 计算模型指标数据
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

    //6 AUC指标
    val roc = binarySummary.roc
    roc.show()
    val AUC = binarySummary.areaUnderROC
    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")

    //6 设置模型阈值
    //不同的阈值，计算不同的F1，然后通过最大的F1找出并重设模型的最佳阈值。
    val fMeasure = binarySummary.fMeasureByThreshold
    fMeasure.show
    //获得最大的F1值
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    //找出最大F1值对应的阈值（最佳阈值）
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure).select("threshold").head().getDouble(0)
    //并将模型的Threshold设置为选择出来的最佳分类阈值
    lrModel.setThreshold(bestThreshold)

    //7 模型保存与加载
    lrModel.save("hdfs://10.49.136.150:9000/user/sunbowhuang/mlv2/lrmodel")
    val load_lrModel = LogisticRegressionModel.load("hdfs://1.1.1.1:9000/user/mlv2/lrmodel")


  }

}
