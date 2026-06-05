from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("check_q1_1").getOrCreate()

df = spark.read.parquet("./q1_task1_output_parquet")

df.printSchema()
df.show(20, truncate=False)
print("row count =", df.count())