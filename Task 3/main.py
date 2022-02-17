import findspark
import pyspark
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import reduce
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.ml.stat import Correlation
from pyspark.sql.types import IntegerType, FloatType, StructType, StructField
from pyspark.ml.feature import VectorAssembler, StandardScaler, VectorIndexer, StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier, NaiveBayes, LogisticRegression, MultilayerPerceptronClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import isnan, when, count, col, max, min, mean, percentile_approx, first, row_number, monotonically_increasing_id, lit, rand
from pyspark.sql import Window

findspark.init()


spark = SparkSession.builder.getOrCreate()
df = spark.read.csv('nuclear_plants_small_dataset.csv', inferSchema=True, header=True).na.fill(0)
#Detecting when a value is messing from the dataset 
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])

print(f"""There are {df.count()} records in the dataset.""")


#Splits the data to all with a normal status 
normal_df = df.where(col('Status') == "Normal")

#finds the max, min, mean and median of each column in the normal dataset 
max_normal_df = normal_df.select([max(c) for c in normal_df.columns]) 
min_normal_df = normal_df.select([min(c) for c in normal_df.columns])
mean_normal_df = normal_df.select([mean(c) for c in normal_df.columns])
median_normal_df = normal_df.select([percentile_approx(i, 0.5)for i in normal_df.columns])

#finds the lower and higher percentiles of each column in the normal dataset 
lower_percentile_normal_df = normal_df.select([percentile_approx(i, 0.25)for i in normal_df.columns])
higher_percentile_normal_df = normal_df.select([percentile_approx(i, 0.75)for i in normal_df.columns])

#Splits the data to all with a abnormal status 
abnormal_df = df.where(col('Status') == "Abnormal")

#finds the max, min, mean and median of each column in the abnormal dataset 
max_abnormal_df = abnormal_df.select([max(c) for c in abnormal_df.columns])
min_abnormal_df = abnormal_df.select([min(c) for c in abnormal_df.columns])
mean_abnormal_df = abnormal_df.select([mean(c) for c in abnormal_df.columns])
median_abnormal_df = abnormal_df.select([percentile_approx(i, 0.5)for i in abnormal_df.columns])

#finds the lower and higher percentiles of each column in the abnormal dataset 
lower_percentile_abnormal_df = abnormal_df.select([percentile_approx(i, 0.25)for i in abnormal_df.columns])
higher_percentile_abnormal_df = abnormal_df.select([percentile_approx(i, 0.75)for i in abnormal_df.columns])

#convers the dataframes to pandas dataframes, this is to print the data in boxplots
pandasNDF = normal_df.toPandas()
pandasADF = abnormal_df.toPandas()

#creates blank lists to store 
normallist = []
abnormallist = []

drop = df.drop("Status")
columns = (drop.columns)
titlelist = ["Status","Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ",'Power_range_sensor_4','Pressure _sensor_1','Pressure _sensor_2','Pressure _sensor_3','Pressure _sensor_4','Vibration_sensor_1','Vibration_sensor_2','Vibration_sensor_3','Vibration_sensor_4']

#Calculates the mode for normal data

for c in columns:
    nm = normal_df.groupBy(c).count().orderBy('count', ascending=False).first()[0]
    normallist.append(nm)

normalrdd = spark.sparkContext.parallelize(normallist)
normalrdd2 = normalrdd.map(lambda c: (c, )).toDF()
nw = Window.orderBy("_1")
na = normalrdd2.withColumn("x4", row_number().over(nw))
nb = na.withColumn("Status", lit("Normal"))
nc = nb.groupBy("Status").pivot("x4").sum("_1")
modenormal = nc.select(col("Status").alias("Status"),col("1").alias("Power_range_sensor_1"),col("2").alias("Power_range_sensor_2"),col("3").alias("Power_range_sensor_3 "),col("4").alias("Power_range_sensor_4"),col("5").alias("Pressure _sensor_1"),col("6").alias("Pressure _sensor_2"),col("7").alias("Pressure _sensor_3"),col("8").alias("Pressure _sensor_4"),col("9").alias("Vibration_sensor_1"),col("10").alias("Vibration_sensor_2"),col("11").alias("Vibration_sensor_3"),col("12").alias("Vibration_sensor_4"))

#Calculates the mode for abnormal data

for p in columns:
    abm = abnormal_df.groupBy(p).count().orderBy('count', ascending=False).first()[0]
    abnormallist.append(abm)

abnormalrdd = spark.sparkContext.parallelize(abnormallist)
abnormalrdd2 = abnormalrdd.map(lambda c: (c, )).toDF()
abnw = Window.orderBy("_1")
abna = abnormalrdd2.withColumn("x4", row_number().over(abnw))
abnb = abna.withColumn("Status", lit("Abnormal"))
abnc = abnb.groupBy("Status").pivot("x4").sum("_1")
modeabnormal = abnc.select(col("Status").alias("Status"),col("1").alias("Power_range_sensor_1"),col("2").alias("Power_range_sensor_2"),col("3").alias("Power_range_sensor_3 "),col("4").alias("Power_range_sensor_4"),col("5").alias("Pressure _sensor_1"),col("6").alias("Pressure _sensor_2"),col("7").alias("Pressure _sensor_3"),col("8").alias("Pressure _sensor_4"),col("9").alias("Vibration_sensor_1"),col("10").alias("Vibration_sensor_2"),col("11").alias("Vibration_sensor_3"),col("12").alias("Vibration_sensor_4"))





labelCol = "Status"


print(f"""There are {df.count()} records in the dataset.""")
columnsdata = df.columns
filteredcol = columnsdata[:-1]
inputCols = filteredcol[1:]
colnumb = [len(inputCols), 15, 15, 2]

labelIndexer = StringIndexer(\
                            inputCol="Status", \
                            outputCol="indexedLabel")

va = VectorAssembler(\
                               inputCols = inputCols, \
                               outputCol = "features") \
                              .setHandleInvalid("skip")

stdScaler = StandardScaler(inputCol="features", \
                        outputCol="scaledFeatures", \
                        withStd=True, \
                        withMean=False)

dt = DecisionTreeClassifier(labelCol="indexedLabel", \
                            featuresCol="features", \
                            impurity="gini")

evaluator = MulticlassClassificationEvaluator(
        labelCol='indexedLabel', 
        predictionCol='prediction', 
        metricName='accuracy')

linear = LinearSVC(maxIter=10, \
        regParam=0.1, \
        featuresCol='scaledFeatures', \
        labelCol='indexedLabel')

multilayerper = MultilayerPerceptronClassifier(labelCol="indexedLabel", \
                                     featuresCol="scaledFeatures", \
                                     maxIter=100, layers=colnumb, \
                                     blockSize=128, \
                                     seed=1234)


randdf = df.orderBy(rand())

#randomizes data
adf = labelIndexer.fit(randdf).transform(randdf)
adf = adf.drop("Status")

#assebles vectors
adf = va.transform(adf)

#trains scaler
scalemodel = stdScaler.fit(adf)
sadf = scalemodel.transform(adf)
#splits data 
trainDF, testDF = sadf.randomSplit([.7, .3], seed=42)

#trains decision tree
dttrained = dt.fit(trainDF)
dtpredict = dttrained.transform(testDF)

#trains the support vector
svmodel = linear.fit(trainDF)
svpredict = svmodel.transform(testDF)

#trains the multilaterperceptron
multimodel = multilayerper.fit(trainDF)
multipredict = multimodel.transform(testDF)

#calculates the accuracy 
multiaccuracy = evaluator.evaluate(multipredict)
vectoraccuracy = evaluator.evaluate(svpredict)
dtaccuracy = evaluator.evaluate(dtpredict)


N_train = trainDF.select("indexedLabel").where(col("indexedLabel") == 0).count()
AB_train = trainDF.select("indexedLabel").where(col("indexedLabel") == 1).count()
N_test = testDF.select("indexedLabel").where(col("indexedLabel") == 0).count()
AB_test = testDF.select("indexedLabel").where(col("indexedLabel") == 1).count()

#decision tree
dtTrueN = dtpredict.where((col('prediction')=='0') & (col('indexedLabel')=='0')).count()
dtFalseN = dtpredict.where((col('prediction')=='1') & (col('indexedLabel')=='0')).count()
dtTrueABN = dtpredict.where((col('prediction')=='1') & (col('indexedLabel')=='1')).count()
dtFalseABN = dtpredict.where((col('prediction')=='0') & (col('indexedLabel')=='1')).count()
#support vector
svTrueN = svpredict.where((col('prediction')=='0') & (col('indexedLabel')=='0')).count()
svFalseN = svpredict.where((col('prediction')=='1') & (col('indexedLabel')=='0')).count()
svTrueABN = svpredict.where((col('prediction')=='1') & (col('indexedLabel')=='1')).count()
svFalseABN = svpredict.where((col('prediction')=='0') & (col('indexedLabel')=='1')).count()
#multilayerperceptron
mtTrueN = multipredict.where((col('prediction')=='0') & (col('indexedLabel')=='0')).count()
mtFalseN = multipredict.where((col('prediction')=='1') & (col('indexedLabel')=='0')).count()
mtTrueABN = multipredict.where((col('prediction')=='1') & (col('indexedLabel')=='1')).count()
mtFalseABN = multipredict.where((col('prediction')=='0') & (col('indexedLabel')=='1')).count()

statlist = ["min","max","mean","median","mode","lower","upper"]
dtSen = (dtTrueN + dtFalseABN)
dtSensitivity = (dtTrueN / dtSen)
dtSpe = (dtTrueABN + dtFalseN)
dtSpecificity = (dtTrueABN / dtSpe)
dtmatrix = spark.createDataFrame([("Normal", dtTrueN, dtFalseN),("Abnormal", dtFalseABN, dtTrueABN)],['Status','True',"False"])

svSen = (svTrueN + svFalseABN)
svSensitivity = (svTrueN / svSen)
svSpe = (svTrueABN + svFalseN)
svSpecificity = (svTrueABN / svSpe)
svmatrix = spark.createDataFrame([("Normal", svTrueN, svFalseN),("Abnormal", svFalseABN, svTrueABN)],['Status','True',"False"])

mtSen = (mtTrueN + mtFalseABN)
mtSensitivity = (mtTrueN / mtSen)
mtSpe = (mtTrueABN + mtFalseN)
mtSpecificity = (mtTrueABN / mtSpe)
mtmatrix = spark.createDataFrame([("Normal", mtTrueN, mtFalseN),("Abnormal", mtFalseABN, mtTrueABN)],['Status','True',"False"])

class boxplots():
    pandasNDF.boxplot(column=["Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ",'Power_range_sensor_4','Pressure _sensor_1','Pressure _sensor_2','Pressure _sensor_3','Pressure _sensor_4','Vibration_sensor_1','Vibration_sensor_2','Vibration_sensor_3','Vibration_sensor_4'],by="Status")
    pandasADF.boxplot(column=["Power_range_sensor_1","Power_range_sensor_2","Power_range_sensor_3 ",'Power_range_sensor_4','Pressure _sensor_1','Pressure _sensor_2','Pressure _sensor_3','Pressure _sensor_4','Vibration_sensor_1','Vibration_sensor_2','Vibration_sensor_3','Vibration_sensor_4'],by="Status")
boxplots()

class showdf():
    print("Normal Max")
    max_normal_df.show()
    print("Normal Min")
    min_normal_df.show()
    print("Normal Mean")
    mean_normal_df.show()
    print("Normal Median")
    median_normal_df.show()
    print("Normal Mode")
    modenormal.show()
    print("Normal Lower Percentile")
    lower_percentile_normal_df.show()
    print("Normal Higher Precentile")
    higher_percentile_normal_df.show()
    print("Abnormal Max")
    max_abnormal_df.show()
    print("Abnormal Min")
    min_abnormal_df.show()
    print("Abnormal Mean")
    mean_abnormal_df.show() 
    print("Abnormal Median") 
    median_abnormal_df.show()
    print("Abnormal Mode")
    modeabnormal.show()
    print("Abnormal Lower Percentile")
    lower_percentile_abnormal_df.show()
    print("Abnormal Higher Precentile")
    higher_percentile_abnormal_df.show()
    boxplots()
    plt.show()  
    print(f"""There are {N_train} rows in the training set, and {N_test} in the test set for the normal dataset""")
    print(f"""There are {AB_train} rows in the training set, and {AB_test} in the test set for the abnormal dataset""")
    print("Accuracy of Decision Tree is = %g"%(dtaccuracy))
    print("Error of Decision Tree is = %g "%(1.0 - dtaccuracy))
    print(f"""Sensitivity of Devision Tree is = {dtSensitivity}, and the Specificity = {dtSpecificity}""")
    dtpredict.select(col("indexedLabel"),col("scaledFeatures"),col("prediction"),col("probability")).show()
    print("Decision Tree Correlation Matric")
    dtmatrix.show()
    print("Accuracy of Multi-layer Perceptron is = %g"%(multiaccuracy))
    print("Error of Multi-layer Perceptron is = %g "%(1.0 - multiaccuracy))
    print(f"""Sensitivity of Multi-Layer Perceptron is = {mtSensitivity}, and the Specificity = {mtSpecificity}""") 
    multipredict.select(col("indexedLabel"),col("scaledFeatures"),col("prediction"),col("probability")).show()
    print("Multi-Layer Perceptron Correlation Matric")
    mtmatrix.show()
    print("Accuracy of Support Vector is = %g"%(vectoraccuracy))
    print("Error of Support Vector is = %g "%(1.0 - vectoraccuracy))
    print(f"""Sensitivity of Support Vector is = {svSensitivity}, and the Specificity = {svSpecificity}""")
    svpredict.select(col("indexedLabel"),col("scaledFeatures"),col("prediction")).show()
    print("Support Vector Correlation Matric")
    svmatrix.show()
showdf()



