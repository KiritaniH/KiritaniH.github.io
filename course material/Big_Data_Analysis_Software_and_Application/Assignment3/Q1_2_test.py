from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("check_q1_2").getOrCreate()
df = spark.read.parquet("./q1_task2_output_parquet")
df.printSchema()
df.show(30, truncate=False)
print(df.count())