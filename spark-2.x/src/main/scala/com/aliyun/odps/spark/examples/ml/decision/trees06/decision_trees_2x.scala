package com.aliyun.odps.spark.examples.ml.decision.trees06

import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{ RandomForestClassificationModel, RandomForestClassifier }
import org.apache.spark.ml.classification.{ DecisionTreeClassifier, DecisionTreeClassificationModel }
import org.apache.spark.ml.classification.{ GBTClassificationModel, GBTClassifier }
import org.apache.spark.ml.evaluation.{ MulticlassClassificationEvaluator, BinaryClassificationEvaluator }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.{ Vector, Vectors }
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.Encoder
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import org.apache.spark.sql.SparkSession

object decision_trees_2x {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().
      master("local")
      .appName("decision_trees_2x")
      .getOrCreate()

    import spark.implicits._

    //1 训练样本准备
    val data = spark.read.format("libsvm").load("hdfs://1.1.1.1:9000/user/sample_libsvm_data.txt")
    data.show

    //2 标签进行索引编号
    val labelIndexer = new StringIndexer().
      setInputCol("label").
      setOutputCol("indexedLabel").
      fit(data)
    // 对离散特征进行标记索引，以用来确定哪些特征是离散特征
    // 如果一个特征的值超过4个以上，该特征视为连续特征，否则将会标记得离散特征并进行索引编号
    val featureIndexer = new VectorIndexer().
      setInputCol("features").
      setOutputCol("indexedFeatures").
      setMaxCategories(4).
      fit(data)

    //3 样本划分
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    //4 训练决策树模型
    val dt = new DecisionTreeClassifier().
      setLabelCol("indexedLabel").
      setFeaturesCol("indexedFeatures")

    //4 训练随机森林模型
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    //4 训练GBDT模型
    val gbt = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    //5 将索引的标签转回原始标签
    val labelConverter = new IndexToString().
      setInputCol("prediction").
      setOutputCol("predictedLabel").
      setLabels(labelIndexer.labels)

    //6 构建Pipeline
    val pipeline1 = new Pipeline().
      setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    val pipeline2 = new Pipeline().
      setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    val pipeline3 = new Pipeline().
      setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

    //7 Pipeline开始训练
    val model1 = pipeline1.fit(trainingData)

    val model2 = pipeline2.fit(trainingData)

    val model3 = pipeline3.fit(trainingData)

    //8 模型测试
    val predictions = model1.transform(testData)
    predictions.show(5)

    //8 测试结果
    predictions.select("predictedLabel", "label", "features").show(5)

    //9 分类指标
    // 正确率
    val evaluator1 = new MulticlassClassificationEvaluator().
      setLabelCol("indexedLabel").
      setPredictionCol("prediction").
      setMetricName("accuracy")
    val accuracy = evaluator1.evaluate(predictions)
    println("Test Error = " + (1.0 - accuracy))
    // f1
    val evaluator2 = new MulticlassClassificationEvaluator().
      setLabelCol("indexedLabel").
      setPredictionCol("prediction").
      setMetricName("f1")
    val f1 = evaluator2.evaluate(predictions)
    println("f1 = " + f1)
    // Precision
    val evaluator3 = new MulticlassClassificationEvaluator().
      setLabelCol("indexedLabel").
      setPredictionCol("prediction").
      setMetricName("weightedPrecision")
    val Precision = evaluator3.evaluate(predictions)
    println("Precision = " + Precision)
    // Recall
    val evaluator4 = new MulticlassClassificationEvaluator().
      setLabelCol("indexedLabel").
      setPredictionCol("prediction").
      setMetricName("weightedRecall")
    val Recall = evaluator4.evaluate(predictions)
    println("Recall = " + Recall)

    // AUC
    val evaluator5 = new BinaryClassificationEvaluator().
      setLabelCol("indexedLabel").
      setRawPredictionCol("prediction").
      setMetricName("areaUnderROC")
    val AUC = evaluator5.evaluate(predictions)
    println("Test AUC = " + AUC)

    // aupr
    val evaluator6 = new BinaryClassificationEvaluator().
      setLabelCol("indexedLabel").
      setRawPredictionCol("prediction").
      setMetricName("areaUnderPR")
    val aupr = evaluator6.evaluate(predictions)
    println("Test aupr = " + aupr)

    //10 决策树打印
    val treeModel = model1.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)

    //11 模型保存与加载
    model1.save("hdfs://1.1.1.1:9000/user//mlv2/dtmodel")
    val load_treeModel = PipelineModel.load("hdfs://1.1.1.1:9000/user//mlv2/dtmodel")

  }

}