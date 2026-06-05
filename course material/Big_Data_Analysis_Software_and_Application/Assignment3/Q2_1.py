from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, to_timestamp, expr

spark = (
    SparkSession.builder
    .appName("Q2_Task1_Activity_Streaming")
    .config("spark.sql.shuffle.partitions", "3")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


input_path = "./Q2_activity_data"
checkpoint_path = "./q2_task1_checkpoint"

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

query = (
    result_df.writeStream
    .format("memory")
    .outputMode("update")
    .queryName("activity_query")
    .option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)
    .start()
)

query.awaitTermination()

spark.sql("""
    SELECT *
    FROM activity_query
    ORDER BY window_start, user
    LIMIT 3
""").show(truncate=False)