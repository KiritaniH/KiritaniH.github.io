from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, expr

spark = (
    SparkSession.builder
    .appName("Q2_Task2_Activity_Streaming")
    .config("spark.sql.shuffle.partitions", "3")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

input_path = "./Q2_activity_data" 
memory_checkpoint_path = "./q2_task2_mem_ckpt"
parquet_checkpoint_path = "./q2_task2_parq_ckpt"
parquet_output_path = "./q2_task2_output_parquet" 

schema = spark.read.json(input_path).schema

stream_df = (
    spark.readStream
    .schema(schema)
    .option("maxFilesPerTrigger", 1)
    .json(input_path)
)

processed_df = (
    stream_df
    .withColumn(
        "event_time",
        expr("to_timestamp(from_unixtime(Creation_Time / 1000000000))")
    )
)

result_df = (
    processed_df
    .withWatermark("event_time", "10 minutes")
    .groupBy(
        window(col("event_time"), "6 minutes", "3 minutes"),
        col("User").alias("user")
    )
    .count()
    .select(
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        col("user"),
        col("count")
    )
)

memory_query = (
    result_df.writeStream
    .format("memory")
    .outputMode("update")
    .queryName("activity_query")
    .option("checkpointLocation", memory_checkpoint_path)
    .trigger(availableNow=True)
    .start()
)

parquet_query = (
    result_df.writeStream
    .format("parquet")
    .outputMode("append")
    .option("path", parquet_output_path)
    .option("checkpointLocation", parquet_checkpoint_path)
    .trigger(availableNow=True)
    .start()
)

memory_query.awaitTermination()
parquet_query.awaitTermination()

spark.sql("""
    SELECT *
    FROM activity_query
    ORDER BY window_start, user
    LIMIT 3
""").show(truncate=False)