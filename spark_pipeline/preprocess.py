from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnull, isnan, when, count
from pyspark.sql.types import FloatType, DoubleType

# Step 1: Initialize Spark session
spark = SparkSession.builder \
    .appName("Smart Supply Chain - Data Exploration") \
    .getOrCreate()

# Step 2: Read the dataset
df = spark.read.csv("data/sales_data.csv", header=True, inferSchema=True)

# Step 3: Print schema
print("Schema:")
df.printSchema()

# Step 4: Show first few rows
print("\nSample Data:")
df.show(5)

# Step 5: Count rows and columns
row_count = df.count()
col_count = len(df.columns)
print(f"\nTotal Rows: {row_count}, Total Columns: {col_count}")

# Step 6: Check for missing values safely
print("\nMissing Value Count per Column:")
missing_counts = df.select([
    count(
        when(
            isnull(col_name) | 
            (isnan(col_name) if df.schema[col_name].dataType in [FloatType(), DoubleType()] else False),
            col_name
        )
    ).alias(col_name)
    for col_name in df.columns
])
missing_counts.show()

# Optional: Stop Spark session
spark.stop()