package com.aliyun.odps.spark.examples.ml.tf

import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, Tokenizer, Word2Vec}
import org.apache.spark.sql.SparkSession
import shapeless.syntax.std.tuple.productTupleOps


object tf_idf_test {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("tf_idf_test")
      .getOrCreate()


     // tf-idf 词频-逆向文档频率
    // 1 样本准备

    // 从 Hive 表中读取商品数据
    val sentenceData = spark.sql("SELECT goods_name FROM dim_goods_list_tmp")


    // 2 通过这样的设置，可以将输入列 sentence 中的文本进行分词，并将分词结果输出到 words 列。
    val tokenizer1 =new Tokenizer().setInputCol("goods_name").setOutputCol("words")
    val wordsData =tokenizer1.transform(sentenceData)
    println("wordsData" +wordsData.show())


    //它会对 words 列的内容进行哈希处理，并将结果以 20 维的特征向量形式存储在 rawFeatures 列中。
    val hashingTF=new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordsData)
    //println("featurizedData" +featurizedData.show())

    //  通过countvectorizer 也可以获得词频向量
    // 将对 rawFeatures 列的数据应用 IDF 算法，并将计算得到的结果输出到 features 列中。IDF 常用于自然语言处理中，用于衡量单词在文档集合中的重要性。
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel =idf.fit(featurizedData)

    // 3 测试及结果展示
    val rescaledData=idfModel.transform(featurizedData)
   //rescaledData.select("label","features").show()




    // 样本准备
    //具体来说，transform 方法会根据 IDF 模型所学习到的参数和规则，对输入的数据 featurizedData 进行相应的计算和处理，并返回转换后的结果。 这个结果通常是对输入数据的特征进行了重新调整或加权，以更好地反映数据中各个特征的重要性或区分度。
    val documentdt =spark.createDataFrame(Seq(
      ("Hi I HEARD ABOUT SPARK").split(" "),
      ("I WISH JAVA COULD USERD CASE CLASS").split("" ),
      ("我是中国人").split(""),
      ("我 爱 北 京 ").split(" ")
    ).map(Tuple1.apply)).toDF("text")

    // documentdt.show()


    // work2vec
    //  Word2Vec 模型对输入的 text 列的文本进行处理，生成维度为 3 的词向量，将单词的最小出现次数设置为 0  并将结果存储在 result 列中。
    val work2vec =new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)

    val model =work2vec.fit(documentdt)

   // 测试及结果展示


    var result = model.transform(documentdt)

    //  result.collect().foreach{   case  Row  }


    // --------------------------特征的变换------------------------//

    // 1. 分词器
    val sentenceDataFrame =spark.createDataFrame(Seq(
      (0.0,"Hi I HEARD ABOUT SPARK"),
      (0.0,"I WISH JAVA COULD USERD CASE CLASS") ,
      (0.0,"我是中国人"),
      (0.0,"我 爱 北 京 ")
    )).toDF("label","sentence")

    val tokenizer =new Tokenizer().setInputCol("sentence").setOutputCol("words")


    val regexTokenizer = new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("\\W")

    val regexTokenizer2 = new RegexTokenizer().setInputCol("sentence").setOutputCol("words").setPattern("\\w+").setGaps(false)

    // udf 计算长度
   // val countTokens = udf {(words:Seq[String]) => words.length}
    // 测试1

    val tokenized=tokenizer.transform(sentenceDataFrame)
  //  tokenized.select("sentence","words").withColumn("tokens",countTokens(col("words"))).show(false)

    // 测试2

    val regexTokenized=regexTokenizer.transform(sentenceDataFrame)
   // regexTokenized.select("sentence","words").withColumn("tokens",countTokens(col("words"))).show(false)

    // 测试3

    val regexTokenized2=regexTokenizer2.transform(sentenceDataFrame)
   // regexTokenized2.select("sentence","words").withColumn("tokens",countTokens(col("words"))).show(false)



  }
}





