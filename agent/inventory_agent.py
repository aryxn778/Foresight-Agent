import requests

def fetch_predicted_demand(input_data: dict) -> float:
    """
    Calls the Flask API to get the predicted demand.
    """
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        response.raise_for_status()
        data = response.json()
        return float(data.get("predicted_demand", 0.0))  # ✅ match app.py
    except Exception as e:
        print(f"[Error] Failed to fetch prediction from API: {e}")
        return 0.0


def dynamic_inventory_order(
    input_data: dict,
    current_inventory: float,
    lead_time_days: int,
    daily_demand_rate: float = None,
    safety_stock: float = 0.1
) -> float:
    """
    Calculates the recommended inventory order using predicted demand.
    """
    forecasted_demand = fetch_predicted_demand(input_data)

    if forecasted_demand == 0:
        print("[Warning] Forecasted demand is 0. Skipping order logic.")
        return 0.0

    if daily_demand_rate is None:
        daily_demand_rate = forecasted_demand / 7.0

    demand_during_lead_time = daily_demand_rate * lead_time_days
    buffer = safety_stock * demand_during_lead_time

    required_stock = demand_during_lead_time + buffer
    order_quantity = required_stock - current_inventory

    return max(0, round(order_quantity, 2))