# restocking_agent.py

from scipy.stats import norm

def calculate_restocking_quantity(
    forecasted_demand: float,
    current_inventory: float,
    lead_time_days: int,
    demand_std_dev: float,
    service_level: float = 0.95
) -> float:
    """
    Calculates the restocking quantity using a service-level-based inventory model.

    Parameters:
    - forecasted_demand (float): Predicted daily demand
    - current_inventory (float): Current inventory level
    - lead_time_days (int): Lead time in days for restocking
    - demand_std_dev (float): Standard deviation of daily demand
    - service_level (float): Desired service level (default is 95%)

    Returns:
    - float: Recommended order quantity (≥ 0)
    """

    # Z-score corresponding to desired service level
    z_score = norm.ppf(service_level)

    # Total demand expected during the lead time
    demand_during_lead_time = forecasted_demand * lead_time_days

    # Safety stock based on variability and service level
    safety_stock = z_score * demand_std_dev * (lead_time_days ** 0.5)

    # Reorder point = expected demand during lead + safety stock
    reorder_point = demand_during_lead_time + safety_stock

    # Order quantity = what we need - what we have
    order_quantity = max(0.0, reorder_point - current_inventory)

    return round(order_quantity, 2)