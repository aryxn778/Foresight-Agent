from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Start SparkSession
spark = SparkSession.builder.appName("CheckModelSchema").getOrCreate()

# Load the model — adjust this path if different
model = PipelineModel.load("model/demand_model")

# Try to inspect the features column
try:
    print("Features column used by model:", model.stages[-1]._call_java("featuresCol"))
    print("Label column used by model:", model.stages[-1]._call_java("labelCol"))
except Exception as e:
    print("Could not fetch model details directly. Error:")
    print(e)

# OPTIONAL: If you want full schema, load a test input DataFrame and show it
# df = spark.read.parquet("path_to_training_data")  # if available
# df.printSchema()

spark.stop()