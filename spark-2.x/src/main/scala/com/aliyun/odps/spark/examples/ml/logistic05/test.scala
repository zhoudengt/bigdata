package com.aliyun.odps.spark.examples.ml.logistic05

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.RegressionEvaluator

/*


使用sacla ml包 写多远线性回归的程序，数据为hive表 ，并给出表数据样例，同时说明表的业务与数据项 以及值的依据
某个预测的特征如何选取
列举几个零售行业的预测案例



your_hive_table 表：
features（逗号分隔的特征值）	label（目标值）
2.5, 1.0, 3.2	            5.6
1.2, 0.5, 2.1	            3.8
3.0, 1.5, 2.8	            6.2
.....	.....
在上述示例中，features 列表示包含多个特征值的列，每个样本的特征值之间用逗号分隔。label 列表示对应的目标值。
对于表的业务、数据项以及值的依据，这完全取决于实际的业务需求和数据情况。以下是一个可能的示例说明：
假设我们正在研究房屋价格的预测问题，your_hive_table 表用于存储与房屋相关的数据。
业务背景：
此表的目的是帮助分析和预测房屋的价格，以便在房地产市场中做出各种决策，例如评估房屋价值、制定合理售价、寻找投资机会等。
数据项：
area（浮点数类型）：房屋的面积，平方米为单位。面积是影响房屋价格的重要因素之一，通常面积越大，价格可能越高。
num_bedrooms（整数类型）：房屋的卧室数量。卧室数量也会对房屋价格产生影响，因为不同家庭对卧室数量的需求不同。
num_bathrooms（整数类型）：房屋的浴室数量。浴室数量多少也会在一定程度上影响房屋的价格和吸引力。
location（字符串类型）：房屋所在的地理位置，例如城市的区域或具体地址。地理位置是决定房屋价格的关键因素之一，不同地区的房价可能有很大差异。
age（整数类型）：房屋的建造年份或年龄。较新的房屋可能在市场上更受欢迎，价格也可能相对较高，但也可能存在一些人喜欢具有历史特色的老房子。
condition（字符串类型）：房屋的状况描述，如良好、一般、需要修缮等。房屋的状况会直接影响其价值和可销售性。
price（浮点数类型）：房屋的价格，这是我们要预测的目标变量。
值的依据：
area 的值通过实际测量房屋的建筑面积得到。
num_bedrooms 和 num_bathrooms 是根据房屋的实际房间数量确定的。
location 按照实际的地址或区域进行记录。
age 基于房屋的建造时间来确定。
condition 通过对房屋的检查和评估得出的描述。
price 可能是根据市场交易数据、评估报告或业主提供的信息等确定的实际价格。
在实际应用中，这些数据项的具体定义和取值依据会根据具体的业务场景和数据来源而有所不同。为了进行多元线性回归分析，需要从 Hive 表中读取这些相关的数据，并将特征数据（如上述示例中的 area、num_bedrooms 等）作为输入 x，目标变量（如 price）作为 y，然后使用类似上述示例中的代码进行多元线性回归建模和分析。
请注意，在实际处理从 Hive 表读取的数据时，可能需要进行一些数据清洗和预处理步骤，以确保数据的质量和适用性，例如处理缺失值、异常值等。此外，多元线性回归的效果还可能受到数据特征之间的相关性、线性关系的强度以及其他因素的影响，可能需要进一步的特征工程、模型选择和评估来获得更准确和可靠的结果。
 */
object test {

  def main(args: Array[String]): Unit = {



    val spark = SparkSession
      .builder()
      .appName("test")
      .getOrCreate()

    // 从 Hive 表读取训练数据  x y
    val trainingData = spark.sql("select count(*) AS feature_column,sum(goods_sale_amount) as label from  df_cb_125003.dwd_trd_film_order_dd  a  WHERE a.dt =20240727 and PRD_DATE=20240201 and order_status=2 and create_date>='2020-01-01' ")

    // 特征工程：将特征列组合成一个向量列
    val assembler = new VectorAssembler()
      .setInputCols(Array("feature_column"))
      .setOutputCol("features")

    val assembledTrainingData = assembler.transform(trainingData)


    // 分离特征和目标变量
    val lrTrainingData = assembledTrainingData.select("features", "label")




    // 创建线性回归模型
    val lr = new LinearRegression()


    // 拟合模型
    val model = lr.fit(lrTrainingData)



    // 从 Hive 表读取测试数据
    val testData = spark.sql("select count(*) AS feature_column,sum(goods_sale_amount) as target_column from  df_cb_125003.dwd_trd_film_order_dd  a  WHERE a.dt =20240727 and PRD_DATE=20240301 and order_status=2 and create_date>='2020-01-01'")

    val assembledTestData = assembler.transform(testData)




    val lrTestData = assembledTestData.select("features", "target_column")

    // 进行预测
    val predictions = model.transform(lrTestData)


    // 评估模型
    val evaluator = new RegressionEvaluator()
      .setLabelCol("target_column")
      .setPredictionCol("prediction")
      .setMetricName("rmse")


    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error = $rmse")

    spark.stop()

  }

}