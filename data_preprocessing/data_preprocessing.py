import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier

# Get the number of executors from the arguments
num_executors = int(sys.argv[1])

# Initialize Spark Session
spark = SparkSession.builder.appName("SparkDataProcesing").config("spark.executor.instances", num_executors).getOrCreate()

# the CSV data into a Spark DataFrame
df = spark.read.csv("Friday-WorkingHours-Morning.pcap_ISCX.csv", header=True, inferSchema=True, sep=',')

## Data Cleaning
# Remove leading spaces from all column names
for column in df.columns:
    df = df.withColumnRenamed(column, column.lstrip())
df = df.drop("Fwd Header Length34", "Fwd Header Length55")

# Replace Infinite Values with NaN
df = df.select([when(col(c) == float('inf'), None)
                 .when(col(c) == float('-inf'), None)
                 .otherwise(col(c)).alias(c) 
                for c in df.columns])

# Drop all NaN
df = df.dropna()

# Encode categorical features
df = df.withColumn("Label", when(df["Label"] == "BENIGN", 0).otherwise(1))

## Feature Selection
# Assemble feature vector
feature_columns = df.columns[:-1] 
assembler = VectorAssembler(inputCols=feature_columns, outputCol="initial_features")
df = assembler.transform(df)

# Prepare random forest classifier for feature selection
rf = RandomForestClassifier(featuresCol="initial_features", labelCol="Label", numTrees=100)
model = rf.fit(df)

# Extract feature importances
importances = model.featureImportances
feature_importances = [(feature, importance) for feature, importance in zip(feature_columns, importances)]

# Rank and select top features
feature_importances.sort(key=lambda x: x[1], reverse=True)  # Sort by importance
top_10_features = feature_importances[:10]  # Select the top 10
top_10_feature_names = [feature for feature, _ in top_10_features] + ["Label"]
df = df.select(*top_10_feature_names)

## Scale the new vector
assembler_top_10 = VectorAssembler(inputCols=top_10_feature_names[:-1], outputCol="top_10_features") 
df = assembler_top_10.transform(df)

scaler = StandardScaler(inputCol="top_10_features", outputCol="scaled_features")
df = scaler.fit(df).transform(df)

# Drop all columns except for 'scaled_features' and 'Label'
df = df.select("scaled_features", "Label")

# Store as CSV file
pandas_df = df.toPandas()
pandas_df.to_csv('data.csv', index=False)

# Stop Spark
spark.stop()