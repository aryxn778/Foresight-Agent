# test_restocking_agent.py

import requests
from restocking_agent import calculate_restocking_quantity

# Step 1: Input features for the Flask API (MUST match app.py schema)
input_data = {
    "Store": 1,
    "Promo": 1,
    "Promo2": 1,
    "SchoolHoliday": 0,
    "CompetitionDistance": 500.0,
    "CompetitionOpenSince": 36.0,
    "Promo2OpenSince": 12.0,
    "year": 2015,
    "month": 5,
    "day": 15,
    "dayOfWeek": 5,
    "lag_1": 1200.0,
    "rolling_mean_3": 1250.0,
    "StateHoliday": "0",
    "StoreType": "a",
    "Assortment": "a",
    "PromoInterval": "Jan,Apr,Jul,Oct"
}

# Step 2: Call Flask API for demand prediction
response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
api_result = response.json()
print("🔁 API Response:", api_result)

# Step 3: Extract predicted demand
predicted_demand = api_result["predicted_demand"]

# Step 4: Define parameters for restocking logic
current_inventory = 700.0         # Current stock level
lead_time_days = 3                # Days to receive new stock
demand_std_dev = 180.0            # Std. deviation of demand (can be historic)
service_level = 0.95              # Desired service level (can be adjusted)

# Step 5: Call the restocking agent
recommended_qty = calculate_restocking_quantity(
    forecasted_demand=predicted_demand,
    current_inventory=current_inventory,
    lead_time_days=lead_time_days,
    demand_std_dev=demand_std_dev,
    service_level=service_level
)

# Step 6: Output the result
print(f"📈 Predicted Demand: {predicted_demand}")
print(f"📦 Recommended Restocking Quantity: {recommended_qty}")