from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


# ------------------------------------------------------------
# 1. Create Spark session
# ------------------------------------------------------------

spark = SparkSession.builder \
    .appName("Q1_User_Clustering_KMeans") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------

DATA_PATH = "Q1_customer_data.csv"

df = spark.read.csv(
    DATA_PATH,
    header=True,
    inferSchema=True
)

print("Original data:")
df.printSchema()
print("Number of original rows:", df.count())
df.show(5, truncate=False)


# ------------------------------------------------------------
# 3. Data cleaning
# ------------------------------------------------------------


df_clean = df.filter(
    (F.col("CustomerID").isNotNull()) &
    (~F.col("InvoiceNo").cast("string").startswith("C")) &
    (F.col("Quantity") > 0) &
    (F.col("UnitPrice") > 0) &
    (F.col("InvoiceDate").isNotNull())
)

print("Number of cleaned rows:", df_clean.count())


# ------------------------------------------------------------
# 4. Convert InvoiceDate to timestamp
# ------------------------------------------------------------


df_clean = df_clean.withColumn(
    "InvoiceTime",
    F.to_timestamp(F.col("InvoiceDate"), "M/d/yyyy H:mm")
)

df_clean = df_clean.filter(F.col("InvoiceTime").isNotNull())


# ------------------------------------------------------------
# 5. Create transaction amount
# ------------------------------------------------------------
# TotalAmount = Quantity * UnitPrice

df_clean = df_clean.withColumn(
    "TotalAmount",
    F.col("Quantity") * F.col("UnitPrice")
)


# ------------------------------------------------------------
# 6. Construct user-level features
# ------------------------------------------------------------

max_time = df_clean.agg(F.max("InvoiceTime").alias("max_time")).collect()[0]["max_time"]

user_df = df_clean.groupBy("CustomerID").agg(
    F.max("InvoiceTime").alias("last_purchase_time"),
    F.countDistinct("InvoiceNo").alias("frequency"),
    F.sum("TotalAmount").alias("monetary"),
    F.sum("Quantity").alias("total_quantity"),
    F.avg("UnitPrice").alias("avg_unit_price"),
    F.countDistinct("StockCode").alias("unique_products"),
    F.count("*").alias("invoice_lines")
)

user_df = user_df.withColumn(
    "recency",
    F.datediff(F.lit(max_time), F.col("last_purchase_time"))
)

user_df = user_df.withColumn(
    "avg_basket_value",
    F.col("monetary") / F.col("frequency")
)

user_df = user_df.withColumn(
    "avg_basket_size",
    F.col("total_quantity") / F.col("frequency")
)

print("Number of users:", user_df.count())

user_df.select(
    "CustomerID",
    "recency",
    "frequency",
    "monetary",
    "total_quantity",
    "avg_unit_price",
    "unique_products",
    "invoice_lines",
    "avg_basket_value",
    "avg_basket_size"
).show(10, truncate=False)


# ------------------------------------------------------------
# 7. Log transformation
# ------------------------------------------------------------


feature_cols = [
    "recency",
    "frequency",
    "monetary",
    "total_quantity",
    "avg_unit_price",
    "unique_products",
    "invoice_lines",
    "avg_basket_value",
    "avg_basket_size"
]

log_feature_cols = []

for c in feature_cols:
    new_c = "log_" + c
    user_df = user_df.withColumn(
        new_c,
        F.log(F.col(c) + F.lit(1.0))
    )
    log_feature_cols.append(new_c)


# ------------------------------------------------------------
# 8. Assemble features
# ------------------------------------------------------------

assembler = VectorAssembler(
    inputCols=log_feature_cols,
    outputCol="raw_features"
)

assembled_df = assembler.transform(user_df)


# ------------------------------------------------------------
# 9. Standardize features
# ------------------------------------------------------------

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)


# ------------------------------------------------------------
# 10. Choose the optimal number of clusters
# ------------------------------------------------------------
# We test K from 2 to 10.
#
# silhouette:
#   A clustering quality score.
#   Larger value means better separated clusters.
#
# trainingCost:
#   Within-cluster sum of squared errors.
#   Smaller value means tighter clusters.
#   This is used for the elbow method.

evaluator = ClusteringEvaluator(
    featuresCol="features",
    predictionCol="prediction",
    metricName="silhouette",
    distanceMeasure="squaredEuclidean"
)

k_results = []

for k in range(2, 11):
    kmeans = KMeans(
        featuresCol="features",
        predictionCol="prediction",
        k=k,
        seed=42,
        maxIter=100
    )

    model = kmeans.fit(scaled_df)
    pred = model.transform(scaled_df)

    silhouette = evaluator.evaluate(pred)
    cost = model.summary.trainingCost

    k_results.append((k, silhouette, cost))

    print(f"K = {k}, silhouette = {silhouette:.4f}, trainingCost = {cost:.2f}")


# Convert K-selection result to Spark DataFrame for display
k_result_df = spark.createDataFrame(
    k_results,
    ["k", "silhouette", "trainingCost"]
)

print("K selection result:")
k_result_df.show()


# ------------------------------------------------------------
# 11. Train final K-means model
# ------------------------------------------------------------

BEST_K = 3

final_kmeans = KMeans(
    featuresCol="features",
    predictionCol="cluster",
    k=BEST_K,
    seed=42,
    maxIter=100
)

final_model = final_kmeans.fit(scaled_df)
clustered_df = final_model.transform(scaled_df)

print("Final clustering result:")
clustered_df.select(
    "CustomerID",
    "cluster",
    "recency",
    "frequency",
    "monetary",
    "total_quantity",
    "avg_unit_price",
    "unique_products",
    "invoice_lines",
    "avg_basket_value",
    "avg_basket_size"
).show(20, truncate=False)


# ------------------------------------------------------------
# 12. Summarize each cluster
# ------------------------------------------------------------

cluster_summary = clustered_df.groupBy("cluster").agg(
    F.count("*").alias("num_customers"),
    F.round(F.avg("recency"), 2).alias("avg_recency"),
    F.round(F.avg("frequency"), 2).alias("avg_frequency"),
    F.round(F.avg("monetary"), 2).alias("avg_monetary"),
    F.round(F.avg("total_quantity"), 2).alias("avg_total_quantity"),
    F.round(F.avg("avg_unit_price"), 2).alias("avg_unit_price"),
    F.round(F.avg("unique_products"), 2).alias("avg_unique_products"),
    F.round(F.avg("avg_basket_value"), 2).alias("avg_basket_value"),
    F.round(F.avg("avg_basket_size"), 2).alias("avg_basket_size")
).orderBy("cluster")

print("Cluster summary:")
cluster_summary.show(truncate=False)


# ------------------------------------------------------------
# 13. Save clustering result
# ------------------------------------------------------------

output_df = clustered_df.select(
    "CustomerID",
    "cluster",
    "recency",
    "frequency",
    "monetary",
    "total_quantity",
    "avg_unit_price",
    "unique_products",
    "invoice_lines",
    "avg_basket_value",
    "avg_basket_size"
)

output_df.coalesce(1).write.csv(
    "q1_user_clustering_result",
    header=True,
    mode="overwrite"
)

print("Clustering result saved to folder: q1_user_clustering_result")




# ------------------------------------------------------------
# 14. Stop Spark session
# ------------------------------------------------------------

spark.stop()