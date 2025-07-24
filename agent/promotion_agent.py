def recommend_promotion(input_data, predicted_demand):
    lag_1 = input_data.get("lag_1", 0)
    rolling_mean_3 = input_data.get("rolling_mean_3", 0)
    promo = input_data.get("Promo", 0)
    school_holiday = input_data.get("SchoolHoliday", 0)
    competition_distance = input_data.get("CompetitionDistance", 9999)
    month = input_data.get("month", 1)
    day_of_week = input_data.get("dayOfWeek", 1)

    recent_trend = (lag_1 + rolling_mean_3) / 2 if (lag_1 + rolling_mean_3) > 0 else 1
    drop_percent = ((predicted_demand - recent_trend) / recent_trend) * 100

    print(f" Predicted Demand: {predicted_demand}")
    print(f" Drop Percent: {drop_percent:.2f}%")

    # --- Early exits ---
    if promo == 1:
        return " Promotion already running"

    if school_holiday == 1 and drop_percent < 15:
        return " No promotion: school holiday likely affecting demand"

    # --- Promotion rules ---
    if drop_percent <= -70:
        return "⚠ Major demand collapse — run aggressive promotion or investigate root cause"
    
    elif drop_percent <= -40:
        return " Recommend flash sale or heavy discount"
    
    elif drop_percent <= -25:
        return " Recommend bundle offer"
    
    elif drop_percent <= -15:
        return " Recommend 10–15% discount"
    
    elif drop_percent <= -5:
        if competition_distance < 500:
            return " Recommend small discount: nearby competitor + slight drop"
        else:
            return " Recommend small discount (5–10%)"
    
    elif drop_percent <= 10:
        return " No promotion needed"
    
    else:
        return " Demand is rising — no promotion"