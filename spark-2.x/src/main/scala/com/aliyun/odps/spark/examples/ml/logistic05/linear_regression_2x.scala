package com.aliyun.odps.spark.examples.ml.logistic05


import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.{ LinearRegression, LinearRegressionModel }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession

object linear_regression_2x {

  def main(args: Array[String]): Unit = {


    val spark = SparkSession
      .builder()
      .appName("linear_regression_2x").getOrCreate()


    import spark.implicits._

    //1 训练样本准备
    val training = spark.read.format("libsvm").load("C:\\PC\\MaxCompute-Spark\\spark-2.x\\src\\main\\resources\\aaa.txt")


    //val hql1 = "select create_date  ,user_id, 'com.jb.gosms.aemoji' as features from df_cb_125003.rec_zy_dim_user_item   limit 100  "
   // val training = spark.sql(hql1)
    training.show(10)



    //2 建立逻辑回归模型
    val lr = new LinearRegression()
      .setMaxIter(100)
      .setRegParam(0.1)
      .setElasticNetParam(0.5)

    println("2 建立逻辑回归模型")

    //2 根据训练样本进行模型训练
    val lrModel = lr.fit(training)
    // 训练的时候数据有问题
    println("2 根据训练样本进行模型训练")
    //2 打印模型信息

    println("2 打印模型信息")
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    println(s"Intercept: ${lrModel.intercept}")

    //4 测试样本
    val test = spark.createDataFrame(Seq(
      (5.601801561245534, Vectors.sparse(10, Array(0,1,2,3,4,5,6,7,8,9), Array(0.6949189734965766,-0.32697929564739403,-0.15359663581829275,-0.8951865090520432,0.2057889391931318,-0.6676656789571533,-0.03553655732400762,0.14550349954571096,0.034600542078191854,0.4223352065067103))),
      (0.2577820163584905, Vectors.sparse(10, Array(0,1,2,3,4,5,6,7,8,9), Array(0.8386555657374337,-0.1270180511534269,0.499812362510895,-0.22686625128130267,-0.6452430441812433,0.18869982177936828,-0.5804648622673358,0.651931743775642,-0.6555641246242951,0.17485476357259122))),
      (1.5299675726687754, Vectors.sparse(10, Array(0,1,2,3,4,5,6,7,8,9), Array(-0.13079299081883855,0.0983382230287082,0.15347083875928424,0.45507300685816965,0.1921083467305864,0.6361110540492223,0.7675261182370992,-0.2543488202081907,0.2927051050236915,0.680182444769418))))).toDF("label", "features")
    test.show

    //5 对模型进行测试
    val test_predict = lrModel.transform(test)
    test_predict.show
    test_predict.select("features", "label", "prediction").collect().foreach {
      case Row(features: Vector, label: Double, prediction: Double) =>
        println(s"($features, $label) -> prediction=$prediction")
    }

    //6 模型摘要
    val trainingSummary = lrModel.summary

    // 每次迭代目标值
    val objectiveHistory = trainingSummary.objectiveHistory
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    //7 模型保存与加载
    //lrModel.save("hdfs://10.49.136.150:9000/user/sunbowhuang/mlv2/lrmodel2")
    // val load_lrModel = LinearRegressionModel.load("hdfs://1.1.1.1:9000/user/sunbowhuang/mlv2/lrmodel2")

  }

}