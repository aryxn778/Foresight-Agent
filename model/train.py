from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
import math

# --- Initialize Spark ---
spark = SparkSession.builder \
    .appName("Rossmann_GBT_Optimized") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

# --- Load Data ---
df = spark.read.parquet("data/rossmann_preprocessed.parquet")
df = df.filter((F.col("Open") == 1) & (F.col("Sales") > 0))

# --- Temporal Features ---
df = df.withColumn("year", F.year("Date")) \
       .withColumn("month", F.month("Date")) \
       .withColumn("day", F.dayofmonth("Date")) \
       .withColumn("dayOfWeek", F.dayofweek("Date")) \
       .withColumn("weekofyear", F.weekofyear("Date"))

# --- Lag and Rolling Features (No leakage) ---
w = Window.partitionBy("Store").orderBy("Date")

df = df.withColumn("lag_1", F.lag("Sales", 1).over(w))
df = df.withColumn("rolling_mean_3", F.avg("Sales").over(w.rowsBetween(-3, -1)))
df = df.fillna({"lag_1": 0, "rolling_mean_3": 0})

# --- Competition & Promo Features ---
df = df.withColumn("CompetitionOpenSince", 
                   12 * (df["year"] - df["CompetitionOpenSinceYear"]) + 
                   (df["month"] - df["CompetitionOpenSinceMonth"]))
df = df.withColumn("CompetitionOpenSince", F.when(F.col("CompetitionOpenSince") < 0, 0)
                                                     .otherwise(F.col("CompetitionOpenSince")))

df = df.withColumn("Promo2OpenSince", 
                   12 * (df["year"] - df["Promo2SinceYear"]) + 
                   (df["weekofyear"] - df["Promo2SinceWeek"]) / 4)
df = df.withColumn("Promo2OpenSince", F.when(F.col("Promo2OpenSince") < 0, 0)
                                           .otherwise(F.col("Promo2OpenSince")))

# --- Log-transform target ---
df = df.withColumn("LogSales", F.log1p("Sales"))

# --- Index categorical variables ---
categorical_cols = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_idx", handleInvalid='keep') 
            for col in categorical_cols]

# --- Features ---
feature_cols = [
    "Store", "Promo", "SchoolHoliday", "CompetitionDistance", 
    "CompetitionOpenSince", "Promo2OpenSince", "Promo2", 
    "year", "month", "day", "dayOfWeek", 
    "lag_1", "rolling_mean_3"
] + [col + "_idx" for col in categorical_cols]

# --- Proper Time-Based Split ---
cutoff_date = "2015-06-01"
train_data = df.filter(F.col("Date") < cutoff_date)
test_data = df.filter(F.col("Date") >= cutoff_date)
train_data = train_data.drop("Date")
test_data = test_data.drop("Date")

# --- Vectorize and scale ---
print("FEATURE COLS USED FOR TRAINING:", feature_cols)
assembler = VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features")
scaler = StandardScaler(inputCol="unscaled_features", outputCol="scaled_features")

# --- GBT Regressor ---
gbt = GBTRegressor(featuresCol="scaled_features", labelCol="LogSales", 
                   maxIter=100, maxDepth=5, subsamplingRate=0.8, seed=42)

# --- Pipeline ---
pipeline = Pipeline(stages=indexers + [assembler, scaler, gbt])

# --- Param Grid (Tuned Down for Speed) ---
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5]) \
    .addGrid(gbt.maxIter, [100]) \
    .build()

# --- TrainValidationSplit ---
tvs = TrainValidationSplit(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=RegressionEvaluator(labelCol="LogSales", predictionCol="prediction", metricName="rmse"),
    trainRatio=0.8,
    seed=42
)

# --- Train model ---
print("Training model...")
model = tvs.fit(train_data)

# --- Predict ---
print("Making predictions...")
predictions = model.transform(test_data)
predictions = predictions.withColumn("prediction_exp", F.expm1("prediction"))

# --- Filter valid predictions ---
preds_filtered = predictions.filter(F.col("Sales") > 0)

# --- Evaluation ---
evaluator_rmse = RegressionEvaluator(labelCol="Sales", predictionCol="prediction_exp", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="Sales", predictionCol="prediction_exp", metricName="mae")

rmse = evaluator_rmse.evaluate(preds_filtered)
mae = evaluator_mae.evaluate(preds_filtered)

# --- MAPE ---
mape_df = preds_filtered.withColumn("ape", F.abs((F.col("Sales") - F.col("prediction_exp")) / F.col("Sales")))
mape = mape_df.agg({"ape": "avg"}).collect()[0][0] * 100

# --- Log-Likelihood ---
log_likelihood_df = preds_filtered.withColumn(
    "ll", -0.5 * F.pow((F.col("Sales") - F.col("prediction_exp")) / 1000, 2)
)
log_likelihood = log_likelihood_df.agg(F.sum("ll")).collect()[0][0]

# --- Output ---
print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ MAPE: {mape:.2f}%")
print(f"✅ Approx Log-Likelihood (scaled): {log_likelihood:.2f}")

# --- Save model ---
model.bestModel.write().overwrite().save("api/model/demand_model")

# --- Done ---
spark.stop()