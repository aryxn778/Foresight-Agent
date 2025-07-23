from pyspark.sql import SparkSession
import pandas as pd

# Initialize Spark
spark = SparkSession.builder.appName("Prepare Final Data").getOrCreate()

# Load feature engineered data
df = spark.read.csv("data/feature_engineered", header=True, inferSchema=True)

# Drop nulls caused by lag/rolling windows
df = df.dropna()

# Convert to Pandas
pdf = df.toPandas()

# Optional: Sort to preserve time order for model training
pdf = pdf.sort_values(by=["product_id", "location_id", "date"])

# Save final dataset
pdf.to_csv("data/final_training.csv", index=False)

spark.stop()