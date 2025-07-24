import requests
from promotion_agent import recommend_promotion

def fetch_predicted_demand(input_data):
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        response.raise_for_status()
        prediction = response.json().get("predicted_demand")
        print(f" API Response: {response.json()}")
        return prediction
    except Exception as e:
        print(f" API Request Failed: {e}")
        return None

# --- Sample Input ---
input_data = {
    "Store": 7,
    "Promo": 0,
    "Promo2": 1,
    "SchoolHoliday": 0,
    "CompetitionDistance": 250.0,
    "CompetitionOpenSince": 36.0,
    "Promo2OpenSince": 12.0,
    "year": 2014,
    "month": 3,
    "day": 11,
    "dayOfWeek": 3,
    "lag_1": 1600.0,
    "rolling_mean_3": 1500.0,
    "StateHoliday": "0",
    "StoreType": "c",
    "Assortment": "a",
    "PromoInterval": "Jan,Apr,Jul,Oct"
}

# 🔧 FORCED TEST VALUE FOR DEMAND (for debugging promotion logic)
DEBUG_MODE = True
FORCED_PREDICTION = 500  # Low value to simulate demand drop

if DEBUG_MODE:
    print(" ⚠ Debug Mode: Using forced predicted demand =", FORCED_PREDICTION)
    predicted_demand = FORCED_PREDICTION
else:
    predicted_demand = fetch_predicted_demand(input_data)

if predicted_demand is not None:
    recommendation = recommend_promotion(input_data, predicted_demand)
    print(f" Final Promotion Recommendation: {recommendation}")
else:
    print(" Could not generate recommendation due to missing prediction.")