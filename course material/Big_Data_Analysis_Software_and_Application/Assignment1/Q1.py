from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, sum, hour, to_timestamp, count
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("Q1").getOrCreate()

# -------------------------
# Q1(1)
# -------------------------

schema = StructType([
    StructField("InvoiceNo", StringType(), True),
    StructField("StockCode", StringType(), True),
    StructField("Description", StringType(), True),
    StructField("Quantity", IntegerType(), True),
    StructField("InvoiceDate", StringType(), True),
    StructField("UnitPrice", DoubleType(), True),
    StructField("CustomerID", IntegerType(), True),
    StructField("Country", StringType(), True)
])

df = spark.read.csv("Q1_data.csv", header=True, schema=schema)

df = df.dropna()
df = df.filter(col("Quantity") > 0)
df = df.filter(col("UnitPrice") > 0)
df = df.filter(~col("InvoiceNo").startswith("C"))

print("Q1(1): First 5 rows of cleaned dataset")
df.show(5)

# =========================================================
# Q1(3) Obs 1:
# =========================================================

df_revenue = df.withColumn("Revenue", col("Quantity") * col("UnitPrice"))
country_sales = (
    df_revenue.groupBy("Country")
    .agg(sum("Revenue").alias("TotalRevenue"))
    .orderBy(col("TotalRevenue").desc())
)

print("Q1(3) Observation 1: Top 10 countries by total revenue")
country_sales.show(10, truncate=False)
top_countries = country_sales.limit(10).toPandas()

plt.figure(figsize=(10, 6))
plt.bar(top_countries["Country"], top_countries["TotalRevenue"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Countries by Total Revenue")
plt.xlabel("Country")
plt.ylabel("Total Revenue")
plt.tight_layout()
plt.savefig("q1_obs1_country_revenue.png")
plt.show()

# =========================================================
# Q1(3) Obs 2:
# =========================================================

df_time = df.withColumn(
    "InvoiceDate_ts",
    to_timestamp(col("InvoiceDate"), "M/d/yyyy H:mm")
)

df_time = df_time.withColumn("Hour", hour(col("InvoiceDate_ts")))

hourly_orders = (
    df_time.groupBy("Hour")
    .agg(count("*").alias("OrderCount"))
    .orderBy("Hour")
)

print("Q1(3) Observation 2: Order distribution by hour")
hourly_orders.show(24, truncate=False)

hourly_pd = hourly_orders.toPandas()

plt.figure(figsize=(10, 6))
plt.plot(hourly_pd["Hour"], hourly_pd["OrderCount"], marker="o")
plt.title("Order Distribution by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Orders")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("q1_obs2_hourly_orders.png")
plt.show()

spark.stop()