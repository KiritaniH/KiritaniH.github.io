from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, window

spark = (
    SparkSession.builder
    .appName("Q1_Task1_IoT_Streaming")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

input_path = "./Q1_IOT" 
output_path = "./q1_task1_output_parquet"
checkpoint_path = "./q1_task1_checkpoint"

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
        "device_type",
        expr("concat_ws('-', slice(split(device_name, '-'), 1, size(split(device_name, '-')) - 1))")
    )
    .withColumn(
        "event_time",
        expr("to_timestamp(from_unixtime(timestamp / 1000))")
    )
)

result_df = (
    processed_df
    .withWatermark("event_time", "5 seconds")
    .groupBy(
        window(col("event_time"), "1 second"),
        col("device_type"),
        col("cn")
    )
    .count()
    .select(
        col("window.start").alias("window_start"),
        col("window.end").alias("window_end"),
        col("device_type"),
        col("cn"),
        col("count")
    )
)

def write_sorted_batch(batch_df, batch_id):
    if batch_df.rdd.isEmpty():
        return

    (
        batch_df
        .orderBy(col("cn").desc())
        .write
        .mode("append")
        .parquet(output_path)
    )

query = (
    result_df.writeStream
    .outputMode("append")
    .option("checkpointLocation", checkpoint_path)
    .trigger(availableNow=True)
    .foreachBatch(write_sorted_batch)
    .start()
)

query.awaitTermination()