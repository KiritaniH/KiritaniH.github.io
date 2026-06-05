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
    .appName("Q3_HousePrice_LightGBM")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

from synapse.ml.lightgbm import LightGBMRegressor

data_dir = "./Q3_house_price_data"
train_path = f"{data_dir}/train.csv"
test_path = f"{data_dir}/test.csv"
output_dir = "./q3_lgbm_predictions_output"

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
    {"numLeaves": 31, "numIterations": 100, "learningRate": 0.05},
    {"numLeaves": 31, "numIterations": 200, "learningRate": 0.05},
    {"numLeaves": 63, "numIterations": 100, "learningRate": 0.05},
    {"numLeaves": 63, "numIterations": 200, "learningRate": 0.03},
]

best_rmse = float("inf")
best_params = None

for params in param_grid:
    lgbm = (
        LightGBMRegressor()
        .setLabelCol(label_col)
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setObjective("regression")
        .setNumLeaves(params["numLeaves"])
        .setNumIterations(params["numIterations"])
        .setLearningRate(params["learningRate"])
    )

    pipeline = Pipeline(stages=indexers + [encoder, imputer, assembler, lgbm])

    model = pipeline.fit(train_split)
    val_pred = model.transform(val_split)
    rmse = evaluator.evaluate(val_pred)

    print(f"Params={params}, Validation RMSE={rmse:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_params = params

print("Best params:", best_params)
print("Best validation RMSE:", best_rmse)

best_lgbm = (
    LightGBMRegressor()
    .setLabelCol(label_col)
    .setFeaturesCol("features")
    .setPredictionCol("prediction")
    .setObjective("regression")
    .setNumLeaves(best_params["numLeaves"])
    .setNumIterations(best_params["numIterations"])
    .setLearningRate(best_params["learningRate"])
)

final_pipeline = Pipeline(stages=indexers + [encoder, imputer, assembler, best_lgbm])
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