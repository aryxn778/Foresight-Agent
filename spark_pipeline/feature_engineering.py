from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, dayofweek, month, lag, avg, when
)
from pyspark.sql.window import Window

# Step 1: Initialize Spark session
spark = SparkSession.builder \
    .appName("Smart Supply Chain - Feature Engineering") \
    .getOrCreate()

# Step 2: Read the preprocessed CSV data
df = spark.read.csv("data/sales_data.csv", header=True, inferSchema=True)

# Step 3: Extract date-based features
df = df.withColumn("day_of_week", dayofweek(col("date"))) \
       .withColumn("month", month(col("date")))

# Step 4: Create lag features (previous day's sales/inventory per product-location)
window_spec = Window.partitionBy("product_id", "location_id").orderBy("date")

df = df.withColumn("prev_units_sold", lag("units_sold").over(window_spec)) \
       .withColumn("prev_inventory_level", lag("inventory_level").over(window_spec))

# Step 5: Add rolling average of units_sold over past 2 days (excluding current)
rolling_window = window_spec.rowsBetween(-2, -1)

df = df.withColumn("rolling_avg_sales", avg("units_sold").over(rolling_window))

# Step 6: Add binary flag for low inventory (e.g. below 50 units)
df = df.withColumn("low_inventory_flag", when(col("inventory_level") < 50, 1).otherwise(0))

# Step 7: Show sample with new features
print("\nData with engineered features:")
df.show(truncate=False)

# Optional: Save the feature-enriched data
df.write.mode("overwrite").option("header", True).csv("data/feature_engineered")

# Stop Spark session
spark.stop()

