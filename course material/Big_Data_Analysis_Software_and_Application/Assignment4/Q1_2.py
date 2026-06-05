from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ------------------------------------------------------------
# 1. Create Spark session
# ------------------------------------------------------------

spark = SparkSession.builder \
    .appName("Q1_TSNE_Visualization") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# ------------------------------------------------------------
# 2. Read clustering result from Q1(1)
# ------------------------------------------------------------

CLUSTER_RESULT_PATH = "q1_user_clustering_result"

df = spark.read.csv(
    CLUSTER_RESULT_PATH,
    header=True,
    inferSchema=True
)

print("Loaded clustering result:")
df.printSchema()
df.show(5, truncate=False)


# ------------------------------------------------------------
# 3. Prepare feature columns
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


# ------------------------------------------------------------
# 4. Log transformation
# ------------------------------------------------------------

log_feature_cols = []

for c in feature_cols:
    new_c = "log_" + c
    df = df.withColumn(
        new_c,
        F.log(F.col(c) + F.lit(1.0))
    )
    log_feature_cols.append(new_c)


# ------------------------------------------------------------
# 5. Assemble and standardize features
# ------------------------------------------------------------

assembler = VectorAssembler(
    inputCols=log_feature_cols,
    outputCol="raw_features"
)

assembled_df = assembler.transform(df)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)


# ------------------------------------------------------------
# 6. Convert to Pandas
# ------------------------------------------------------------

from pyspark.ml.functions import vector_to_array

scaled_df = scaled_df.withColumn(
    "features_array",
    vector_to_array(F.col("features"))
)

pd_df = scaled_df.select(
    "CustomerID",
    "cluster",
    "features_array",
    "recency",
    "frequency",
    "monetary",
    "total_quantity",
    "avg_unit_price",
    "unique_products",
    "avg_basket_value",
    "avg_basket_size"
).toPandas()

X = np.vstack(pd_df["features_array"].values)

print("Feature matrix shape:", X.shape)


# ------------------------------------------------------------
# 7. Run t-SNE
# ------------------------------------------------------------

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(X)

pd_df["tsne_1"] = X_tsne[:, 0]
pd_df["tsne_2"] = X_tsne[:, 1]


# ------------------------------------------------------------
# 8. Plot t-SNE result
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    pd_df["tsne_1"],
    pd_df["tsne_2"],
    c=pd_df["cluster"],
    cmap="tab10",
    alpha=0.75,
    s=20
)

plt.title("t-SNE Visualization of Customer Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

cbar = plt.colorbar(scatter)
cbar.set_label("Cluster")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("q1_tsne_customer_clusters.png", dpi=300)
plt.show()


# ------------------------------------------------------------
# 9. Save t-SNE result
# ------------------------------------------------------------

pd_df.to_csv(
    "q1_tsne_customer_clusters.csv",
    index=False
)

print("t-SNE figure saved as: q1_tsne_customer_clusters.png")
print("t-SNE data saved as: q1_tsne_customer_clusters.csv")


# ------------------------------------------------------------
# 10. Stop Spark session
# ------------------------------------------------------------

spark.stop()