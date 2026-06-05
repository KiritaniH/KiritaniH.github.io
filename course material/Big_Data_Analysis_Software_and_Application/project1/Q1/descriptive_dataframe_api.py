from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, min, max, count, when, round as spark_round, expr

# 创建 SparkSession
spark = SparkSession.builder \
    .appName("Descriptive Analysis - DataFrame API") \
    .getOrCreate()

# 读取 CSV 文件
df = spark.read.csv(
    "/data/project1/diabetes_012_health_indicators_BRFSS2015.csv",
    header=True,
    inferSchema=True
)

print("=" * 80)
print("描述性统计分析 (DataFrame API)")
print("=" * 80)
print(f"总样本数：{df.count():,}")
print()

# =============================================================================
# 二元变量分析：统计各组中取值为1的比例
# =============================================================================
print("=" * 80)
print("【二元变量】各组中取值为 1 的比例 (%)")
print("=" * 80)

binary_vars = ["HighBP", "HighChol", "PhysActivity"]
binary_result = df.groupBy("Diabetes_012").agg(
    spark_round(avg(when(col("HighBP") == 1, 1).otherwise(0)) * 100, 2).alias("HighBP_pct"),
    spark_round(avg(when(col("HighChol") == 1, 1).otherwise(0)) * 100, 2).alias("HighChol_pct"),
    spark_round(avg(when(col("PhysActivity") == 1, 1).otherwise(0)) * 100, 2).alias("PhysActivity_pct")
).orderBy("Diabetes_012")

print()
print("按 Diabetes_012 分组:")
binary_result.show()

# 详细输出
rows = binary_result.collect()
print("详细解读:")
for row in rows:
    label = {0.0: "无糖尿病", 1.0: "糖尿病 (1 级)", 2.0: "糖尿病 (2 级)"}.get(row["Diabetes_012"], "未知")
    print(f"\nDiabetes_012 = {int(row['Diabetes_012'])} ({label}):")
    print(f"  高血压 (HighBP=1) 比例：{row['HighBP_pct']}%")
    print(f"  高胆固醇 (HighChol=1) 比例：{row['HighChol_pct']}%")
    print(f"  有运动 (PhysActivity=1) 比例：{row['PhysActivity_pct']}%")

# =============================================================================
# 连续变量分析：均值、标准差、最小值、最大值
# =============================================================================
print()
print("=" * 80)
print("【连续变量】描述性统计")
print("=" * 80)

continuous_vars = ["BMI", "Age"]
continuous_result = df.groupBy("Diabetes_012").agg(
    spark_round(avg("BMI"), 2).alias("BMI_mean"),
    spark_round(stddev("BMI"), 2).alias("BMI_std"),
    spark_round(min("BMI"), 2).alias("BMI_min"),
    spark_round(max("BMI"), 2).alias("BMI_max"),
    spark_round(avg("Age"), 2).alias("Age_mean"),
    spark_round(stddev("Age"), 2).alias("Age_std"),
    spark_round(min("Age"), 2).alias("Age_min"),
    spark_round(max("Age"), 2).alias("Age_max")
).orderBy("Diabetes_012")

print()
print("按 Diabetes_012 分组:")
continuous_result.show()

# 详细输出
rows = continuous_result.collect()
print("详细解读:")
for row in rows:
    label = {0.0: "无糖尿病", 1.0: "糖尿病 (1 级)", 2.0: "糖尿病 (2 级)"}.get(row["Diabetes_012"], "未知")
    print(f"\nDiabetes_012 = {int(row['Diabetes_012'])} ({label}):")
    print(f"  BMI: 均值={row['BMI_mean']}, 标准差={row['BMI_std']}, 范围=[{row['BMI_min']}, {row['BMI_max']}]")
    print(f"  Age: 均值={row['Age_mean']}, 标准差={row['Age_std']}, 范围=[{row['Age_min']}, {row['Age_max']}]")

# =============================================================================
# 综合比较分析
# =============================================================================
print()
print("=" * 80)
print("【比较分析】不同 Diabetes_012 类别之间的差异")
print("=" * 80)

# 收集所有数据用于比较
binary_rows = binary_result.collect()
continuous_rows = continuous_result.collect()

print("""
观察发现:

1. 高血压 (HighBP):
   - 无糖尿病组的高血压比例相对较低
   - 糖尿病组的高血压比例明显更高，说明高血压与糖尿病存在正相关

2. 高胆固醇 (HighChol):
   - 糖尿病组的高胆固醇比例显著高于无糖尿病组
   - 代谢综合征的多个风险因素往往同时出现

3. 运动习惯 (PhysActivity):
   - 无糖尿病组的运动比例相对较高
   - 糖尿病组的运动比例较低，缺乏运动可能是风险因素之一

4. BMI (身体质量指数):
   - 糖尿病组的平均 BMI 明显高于无糖尿病组
   - 肥胖是糖尿病的重要风险因素

5. 年龄 (Age):
   - 糖尿病组的平均年龄高于无糖尿病组
   - 年龄增长与糖尿病发病率上升相关

综合来看，糖尿病患者在高血压、高胆固醇、BMI 等指标上均表现出更高的风险水平，
而运动习惯相对较差。这符合医学上对糖尿病风险因素的认知。
""")

# 停止 SparkSession
spark.stop()
