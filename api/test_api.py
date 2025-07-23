import requests
import json

url = "http://127.0.0.1:5000/predict"
payload = {
    "Store": 1,
    "Promo": 1,
    "SchoolHoliday": 0,
    "CompetitionDistance": 250.0,
    "CompetitionOpenSince": 24,
    "Promo2OpenSince": 10,
    "Promo2": 1,
    "year": 2015,
    "month": 5,
    "day": 15,
    "dayOfWeek": 6,
    "lag_1": 5500.0,
    "rolling_mean_3": 5300.0,
    "StateHoliday_idx": 0.0,
    "StoreType_idx": 1.0,
    "Assortment_idx": 0.0,
    "PromoInterval_idx": 2.0
}

response = requests.post(url, json=payload)
print("Response:", response.json())