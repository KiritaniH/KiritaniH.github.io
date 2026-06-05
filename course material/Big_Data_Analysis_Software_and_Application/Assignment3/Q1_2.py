from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, window, sum as spark_sum, when

spark = (
    SparkSession.builder
    .appName("Q1_Task2_IoT_Streaming")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

input_path = "./Q1_IOT"
output_path = "./q1_task2_output_parquet"
checkpoint_path = "./q1_task2_checkpoint"

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
        expr("to_timestamp(from_unixtime(timestamp / 1000))")
    )
)

flagged_df = (
    processed_df
    .withColumn(
        "low_battery_flag",
        when(col("battery_level") <= 5, 1).otherwise(0)
    )
)

result_df = (
    flagged_df
    .withWatermark("event_time", "5 seconds")
    .groupBy(
        window(col("event_time"), "1 second"),
        col("cn")
    )
    .agg(
        spark_sum(col("low_battery_flag")).alias("low_battery_device_count")
    )
    .select(
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        col("cn"),
        col("low_battery_device_count")
    )
)

query = (
    result_df.writeStream
    .format("parquet")
    .outputMode("append")
    .option("path", output_path)
    .option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)
    .start()
)

query.awaitTermination()