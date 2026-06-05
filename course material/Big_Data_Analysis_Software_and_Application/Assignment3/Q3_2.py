from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, trim
from pyspark.sql.types import (
    IntegerType, LongType, DoubleType, FloatType, ShortType, DecimalType, StringType
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.evaluation import RegressionEvaluator

spark = (
    SparkSession.builder
    .appName("Q3_HousePrice_XGBoost")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

try:
    from xgboost.spark import SparkXGBRegressor
except ImportError:
    try:
        from sparkxgb import XGBoostRegressor as SparkXGBRegressor
    except ImportError:
        raise ImportError(
            "XGBoost PySpark package not found. "
            "Please install xgboost with Spark support first."
        )

data_dir = "./Q3_house_price_data"
train_path = f"{data_dir}/train.csv"
test_path = f"{data_dir}/test.csv"
output_dir = "./q3_xgb_predictions_output"

label_col = "SalePrice"
id_col = "Id"

train_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(train_path)
)

test_df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(test_path)
)

train_schema_dict = {field.name: field.dataType for field in train_df.schema.fields}

for c in train_df.columns:
    s = trim(col(c).cast("string"))
    train_df = train_df.withColumn(
        c,
        when(s.isNull() | s.isin("", "NA"), None).otherwise(col(c))
    )

for c in test_df.columns:
    s = trim(col(c).cast("string"))
    test_df = test_df.withColumn(
        c,
        when(s.isNull() | s.isin("", "NA"), None).otherwise(s)
    )

for c in test_df.columns:
    if c in train_schema_dict:
        test_df = test_df.withColumn(c, col(c).cast(train_schema_dict[c]))

train_df = train_df.filter(col(label_col).isNotNull())

numeric_types = (
    IntegerType, LongType, DoubleType, FloatType, ShortType, DecimalType
)

feature_cols = [c for c in train_df.columns if c != label_col]

numeric_cols = [
    field.name for field in train_df.schema.fields
    if field.name in feature_cols and isinstance(field.dataType, numeric_types)
]

categorical_cols = [
    field.name for field in train_df.schema.fields
    if field.name in feature_cols and isinstance(field.dataType, StringType)
]

numeric_cols = [c for c in numeric_cols if c in test_df.columns]
categorical_cols = [c for c in categorical_cols if c in test_df.columns]

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=[f"{c}_imputed" for c in numeric_cols]
).setStrategy("median")

imputed_numeric_cols = [f"{c}_imputed" for c in numeric_cols]

indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=f"{c}_idx",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

indexed_categorical_cols = [f"{c}_idx" for c in categorical_cols]

encoder = OneHotEncoder(
    inputCols=indexed_categorical_cols,
    outputCols=[f"{c}_ohe" for c in categorical_cols],
    handleInvalid="keep"
)

encoded_categorical_cols = [f"{c}_ohe" for c in categorical_cols]

assembler_inputs = imputed_numeric_cols + encoded_categorical_cols

assembler = VectorAssembler(
    inputCols=assembler_inputs,
    outputCol="features",
    handleInvalid="keep"
)

train_split, val_split = train_df.randomSplit([0.8, 0.2], seed=42)

evaluator = RegressionEvaluator(
    labelCol=label_col,
    predictionCol="prediction",
    metricName="rmse"
)

param_grid = [
    {"max_depth": 4, "eta": 0.05, "num_round": 200},
    {"max_depth": 6, "eta": 0.05, "num_round": 200},
    {"max_depth": 6, "eta": 0.1,  "num_round": 150},
    {"max_depth": 8, "eta": 0.05, "num_round": 150},
]

best_rmse = float("inf")
best_params = None

for params in param_grid:

    xgb = SparkXGBRegressor(
        features_col="features",
        label_col=label_col,
        prediction_col="prediction",
        objective="reg:squarederror",
        max_depth=params["max_depth"],
        eta=params["eta"],
        num_round=params["num_round"]
    )

    pipeline = Pipeline(stages=indexers + [encoder, imputer, assembler, xgb])

    model = pipeline.fit(train_split)
    val_pred = model.transform(val_split)
    rmse = evaluator.evaluate(val_pred)

    print(f"Params={params}, Validation RMSE={rmse:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params

print("Best params:", best_params)
print("Best validation RMSE:", best_rmse)

best_xgb = SparkXGBRegressor(
    features_col="features",
    label_col=label_col,
    prediction_col="prediction",
    objective="reg:squarederror",
    max_depth=best_params["max_depth"],
    eta=best_params["eta"],
    num_round=best_params["num_round"]
)

final_pipeline = Pipeline(stages=indexers + [encoder, imputer, assembler, best_xgb])
final_model = final_pipeline.fit(train_df)

test_pred = final_model.transform(test_df)

submission_df = (
    test_pred
    .select(
        col(id_col).alias("id"),
        when(col("prediction") < 0, lit(0.0))
        .otherwise(col("prediction"))
        .alias("SalePrice")
    )
)

submission_df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_dir)

print("Prediction finished.")
print("Output directory:", output_dir)
print("Please copy the generated part-*.csv to predictions.csv")