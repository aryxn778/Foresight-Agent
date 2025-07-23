from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

def load_data():
    spark = SparkSession.builder.appName("RossmannData").getOrCreate()

    # Load preprocessed Parquet file
    df = spark.read.parquet("data/rossmann_preprocessed.parquet")

    # --- Step 1: Convert date to numeric features ---
    df = df.withColumn("Year", year("Date"))
    df = df.withColumn("Month", month("Date"))
    df = df.withColumn("Day", dayofmonth("Date"))
    df = df.drop("Date")  # drop original Date column

    # --- Step 2: Encode categorical columns ---
    categorical_cols = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid='keep')
        for col in categorical_cols
    ]

    # --- Step 3: Assemble final feature vector ---
    feature_cols = [
        "Store", "DayOfWeek", "Promo", "SchoolHoliday",
        "Year", "Month", "Day",
        "StateHoliday_indexed", "StoreType_indexed", "Assortment_indexed", "PromoInterval_indexed"
    ]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    pipeline = Pipeline(stages=indexers + [assembler])
    pipeline_model = pipeline.fit(df)
    df_transformed = pipeline_model.transform(df)

    # --- Step 4: Final dataset with label and features only ---
    final_df = df_transformed.select("features", "Sales").withColumnRenamed("Sales", "label")

    return final_df