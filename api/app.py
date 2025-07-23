from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
import traceback
import math

app = Flask(__name__)

# --- Start Spark Session ---
spark = SparkSession.builder \
    .appName("DemandForecastAPI") \
    .getOrCreate()

# --- Load Trained Pipeline ---
model_path = "model/demand_model"
model = PipelineModel.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📥 Received input JSON:", data)

        # Convert incoming dict to Spark DataFrame
        df = spark.createDataFrame([data])

        # Reapply feature engineering
        df = df.withColumn("Promo2OpenSince", F.when(F.col("Promo2OpenSince").isNull(), 0.0).otherwise(F.col("Promo2OpenSince")))
        df = df.withColumn("CompetitionOpenSince", F.when(F.col("CompetitionOpenSince").isNull(), 0.0).otherwise(F.col("CompetitionOpenSince")))

        # Ensure all numerical types
        df = df.withColumn("lag_1", df["lag_1"].cast("double"))
        df = df.withColumn("rolling_mean_3", df["rolling_mean_3"].cast("double"))
        df = df.withColumn("CompetitionDistance", df["CompetitionDistance"].cast("double"))
        df = df.withColumn("Promo2OpenSince", df["Promo2OpenSince"].cast("double"))
        df = df.withColumn("CompetitionOpenSince", df["CompetitionOpenSince"].cast("double"))

        # --- Predict ---
        prediction_df = model.transform(df)
        prediction = prediction_df.select("prediction").collect()[0][0]

        # Convert from log1p back to original scale
        predicted_sales = float(round(math.expm1(prediction), 2))

        print("✅ Predicted Demand:", predicted_sales)
        return jsonify({"predicted_demand": predicted_sales})  # ✅ correct key!

    except Exception as e:
        print("❌ Exception occurred:", e)
        print(traceback.format_exc())
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        })

if __name__ == '__main__':
    app.run(debug=True)