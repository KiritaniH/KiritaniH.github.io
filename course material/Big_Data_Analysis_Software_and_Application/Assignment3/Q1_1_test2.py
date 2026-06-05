from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, window, max as spark_max

spark = SparkSession.builder.appName("check_q1_1_closed_windows").getOrCreate()

result = spark.read.parquet("./q1_task1_output_parquet")

print("=== Result Row Count ===")
print(result.count())

raw = (
    spark.read.json("./Q1_IOT")
    .withColumn(
        "device_type",
        expr("concat_ws('-', slice(split(device_name, '-'), 1, size(split(device_name, '-')) - 1))")
    )
    .withColumn(
        "event_time",
        expr("to_timestamp(from_unixtime(timestamp / 1000))")
    )
)

max_event_time = raw.select(spark_max("event_time").alias("mx")).collect()[0]["mx"]
print("max_event_time =", max_event_time)

expected_all = (
    raw
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

expected_closed = expected_all.filter(
    col("window_end") <= expr(f"timestamp'{max_event_time}' - interval 5 seconds")
)

print("=== Expected Closed Row Count ===")
print(expected_closed.count())

diff1 = expected_closed.exceptAll(result)
diff2 = result.exceptAll(expected_closed)

print("expected_closed - result =", diff1.count())
print("result - expected_closed =", diff2.count())