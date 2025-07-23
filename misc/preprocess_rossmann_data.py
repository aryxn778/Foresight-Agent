from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    year, month, dayofmonth, dayofweek, weekofyear,
    when, col, isnan, to_date  # 🔧 Added to_date
)
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Start Spark
spark = SparkSession.builder.appName("RossmannPreprocessing").getOrCreate()

# Load merged dataset
df = spark.read.csv("data/train.csv", header=True, inferSchema=True)
df_store = spark.read.csv("data/store.csv", header=True, inferSchema=True)
df = df.join(df_store, on="Store", how="left")

# 🔧 Parse Date using correct format
df = df.withColumn("Date", to_date("Date", "M/d/yyyy"))

# ---- FEATURE ENGINEERING ---- #

# Extract temporal features
df = (
    df.withColumn("Year", year("Date"))
      .withColumn("Month", month("Date"))
      .withColumn("Day", dayofmonth("Date"))
      .withColumn("DayOfWeek", dayofweek("Date"))
      .withColumn("WeekOfYear", weekofyear("Date"))
)

# Fill NA values
df = df.fillna({
    "CompetitionDistance": 0,
    "CompetitionOpenSinceMonth": 0,
    "CompetitionOpenSinceYear": 0,
    "Promo2SinceWeek": 0,
    "Promo2SinceYear": 0,
    "PromoInterval": "None"
})

# ---- ENCODING CATEGORICALS ---- #

categorical_cols = ["StoreType", "Assortment", "StateHoliday", "PromoInterval"]
indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col + "_Index", outputCol=col + "_OHE") for col in categorical_cols]

# ---- ASSEMBLE FEATURES ---- #

feature_cols = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday",
    "CompetitionDistance", "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
    "Year", "Month", "Day", "WeekOfYear"
] + [col + "_OHE" for col in categorical_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# ---- BUILD PIPELINE ---- #

pipeline = Pipeline(stages=indexers + encoders + [assembler])
pipeline_model = pipeline.fit(df)
df_preprocessed = pipeline_model.transform(df)

# ---- FINAL PREVIEW ---- #

df_preprocessed.select("features", "Sales").show(3, truncate=False)

# ---- SAVE (Optional) ---- #
df_preprocessed.write.mode("overwrite").parquet("data/rossmann_preprocessed.parquet")

spark.stop()