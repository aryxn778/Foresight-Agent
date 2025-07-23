from inventory_agent import dynamic_inventory_order

# Step 1: Prepare full input data
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

# Step 2: Set inventory conditions
current_inventory = 500.0
lead_time_days = 2

# Step 3: Call the agent
recommended_qty = dynamic_inventory_order(
    input_data=input_data,
    current_inventory=current_inventory,
    lead_time_days=lead_time_days,
)

# Step 4: Display result
print(f"✅ Recommended Order Quantity: {recommended_qty}")