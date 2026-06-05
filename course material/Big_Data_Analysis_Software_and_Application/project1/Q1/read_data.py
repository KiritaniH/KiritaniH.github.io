from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("Diabetes Data Analysis") \
    .getOrCreate()

# 读取 CSV 文件
df = spark.read.csv(
    "/data/project1/diabetes_012_health_indicators_BRFSS2015.csv",
    header=True,
    inferSchema=True
)

# 展示 schema
print("=" * 50)
print("DataFrame Schema:")
print("=" * 50)
df.printSchema()

# 展示前几行数据
print("\n" + "=" * 50)
print("Preview (first 5 rows):")
print("=" * 50)
df.show(5)

# 基本信息
print("\n" + "=" * 50)
print("Basic Info:")
print("=" * 50)
print(f"Total rows: {df.count()}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {df.columns}")

# 停止 SparkSession
spark.stop()
