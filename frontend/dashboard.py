import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt

from agent.promotion_agent import recommend_promotion
from agent.restocking_agent import calculate_restocking_quantity
from agent.inventory_agent import dynamic_inventory_order

# --- Enhanced Modern CSS with Animations ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Solid off white bg */
    .stApp {
        background: #eff6e0!important;
        background-size: 400% 400%;
        animation: gradientFlow 12s ease infinite;
    }
    
    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main {
        background: transparent !important;
        padding: 0 !important;
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        margin: 20px;
        padding: 30px;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: slideUp 1s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Enhanced Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
        animation: slideLeft 0.8s ease-out;
    }
    
    @keyframes slideLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .css-1d391kg .css-17eq0hr {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    .css-1d391kg label {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar Expander Styling */
    .css-1d391kg .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #000000 !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .css-1d391kg .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 0 0 10px 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-top: none !important;
    }
    
    /* Input Fields */
    .stSelectbox select, .stNumberInput input, .stTextInput input, .stSlider {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        color: #000000 !important;
        backdrop-filter: blur(5px) !important;
    }
    
    /* Header Animation */
    .dashboard-header {
        text-align: center;
        margin-bottom: 30px;
        padding: 20px 0;
        animation: fadeIn 1.2s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .dashboard-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #000000 0%, #333333 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
        animation: titleGlow 3s ease-in-out infinite alternate;
        text-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
    }
    
    @keyframes titleGlow {
        from { 
            filter: brightness(1);
            text-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
        }
        to { 
            filter: brightness(1.2);
            text-shadow: 0 0 30px rgba(0, 0, 0, 0.8);
        }
    }
    
    .dashboard-subtitle {
        color: #000000 !important;
        font-size: 1.2rem;
        font-weight: 400;
        line-height: 1.6;
        text-shadow: 0 2px 10px rgba(255, 255, 255, 0.3);
    }
    
    /* Animated Button */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 15px 40px !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
        transition: all 0.3s ease !important;
        animation: buttonPulse 2s ease-in-out infinite alternate;
        min-height: 60px !important;
    }
    
    @keyframes buttonPulse {
        from { 
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
            transform: translateY(0);
        }
        to { 
            box-shadow: 0 12px 35px rgba(59, 130, 246, 0.6);
            transform: translateY(-2px);
        }
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 15px 40px rgba(59, 130, 246, 0.7) !important;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
    }
    
    /* Metric Cards with Animation */
    .metric-display {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.1) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.4);
        animation: cardSlideIn 1s ease-out;
        transition: all 0.4s ease;
    }
    
    .metric-display:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.6);
    }
    
    @keyframes cardSlideIn {
        from {
            opacity: 0;
            transform: translateY(30px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    
    /* Enhanced Text */
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #000000 0%, #333333 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: valueShimmer 2s ease-in-out infinite alternate;
    }
    
    @keyframes valueShimmer {
        from { opacity: 0.8; }
        to { opacity: 1; }
    }
    
    /* Chart Container */
    .chart-wrapper {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        margin: 25px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: chartAppear 1.2s ease-out;
    }
    
    @keyframes chartAppear {
        from {
            opacity: 0;
            transform: scale(0.9) rotateY(10deg);
        }
        to {
            opacity: 1;
            transform: scale(1) rotateY(0deg);
        }
    }
    
    /* Recommendation Boxes */
    .recommendation-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.1) 100%);
        backdrop-filter: blur(15px);
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border-left: 4px solid #60a5fa;
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: recommendationSlide 1.3s ease-out;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
    }
    
    @keyframes recommendationSlide {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .recommendation-title {
        color: #000000 !important;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 15px;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.3);
    }
    
    .recommendation-content {
        color: #000000 !important;
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(5px);
    }
    
    /* Success/Info Messages */
    .stAlert {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #000000 !important;
    }
    
    /* Streamlit Metrics - Force center alignment */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        margin: 10px 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        animation: metricFade 1.5s ease-out !important;
        text-align: center !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        min-height: 120px !important;
    }
    
    div[data-testid="metric-container"] > div {
        text-align: center !important;
        width: 100% !important;
    }
    
    div[data-testid="metric-container"] label {
        color: #000000 !important;
        font-weight: 500 !important;
        text-align: center !important;
        display: block !important;
        width: 100% !important;
        margin-bottom: 8px !important;
        font-size: 0.9rem !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        text-align: center !important;
        display: block !important;
        width: 100% !important;
        line-height: 1.2 !important;
    }
    
    /* Remove old metric container styling */
    .metric-container {
        background: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    @keyframes metricFade {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Divider */
    .animated-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #60a5fa, #3b82f6, #60a5fa, transparent);
        border: none;
        margin: 30px 0;
        border-radius: 2px;
        animation: dividerGlow 3s ease-in-out infinite;
    }
    
    @keyframes dividerGlow {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Black text for better contrast */
    .stMarkdown, .stText, p, span, div {
        color: #000000 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        text-shadow: 0 2px 4px rgba(255, 255, 255, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Header with enhanced styling
st.markdown("""
<div class="dashboard-header">
    <h1 class="dashboard-title">Smart Supply Chain Agent Dashboard</h1>
    <p class="dashboard-subtitle">
        Intelligent demand forecasting and supply chain optimization powered by AI.<br>
        Configure your store parameters and get actionable insights for promotions, restocking, and inventory management.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar with original names and always expanded sections
st.sidebar.markdown("### 📊 Store Input Features")

# Store Basic Details - Always Expanded
with st.sidebar.expander("Store Details", expanded=True):
    store = st.number_input("Store", min_value=1, max_value=100, value=1)
    store_type = st.selectbox("StoreType", ["a", "b", "c", "d"])
    assortment = st.selectbox("Assortment", ["a", "b", "c"])

# Promotions - Always Expanded
with st.sidebar.expander("Promotions", expanded=True):
    promo = st.selectbox("Promo", [0, 1])
    promo2 = st.selectbox("Promo2", [0, 1])
    promo_interval = st.text_input("PromoInterval", "Jan,Apr,Jul,Oct")
    promo2_open_since = st.number_input("Promo2OpenSince", min_value=0.0, value=10.0)

# Date & External Factors - Always Expanded
with st.sidebar.expander("Date & External Factors", expanded=True):
    year = st.number_input("Year", min_value=2010, max_value=2030, value=2015)
    month = st.number_input("Month", min_value=1, max_value=12, value=5)
    day = st.number_input("Day", min_value=1, max_value=31, value=15)
    day_of_week = st.number_input("DayOfWeek", min_value=1, max_value=7, value=6)
    school_holiday = st.selectbox("SchoolHoliday", [0, 1])
    state_holiday = st.selectbox("StateHoliday", ["0", "a", "b", "c"])

# Competition - Always Expanded
with st.sidebar.expander("Competition", expanded=True):
    competition_distance = st.number_input("CompetitionDistance", min_value=0.0, value=250.0)
    competition_open_since = st.number_input("CompetitionOpenSince", min_value=0.0, value=24.0)

# Historical Data - Always Expanded
with st.sidebar.expander("Historical Data", expanded=True):
    lag_1 = st.number_input("lag_1", min_value=0.0, value=5500.0)
    rolling_mean_3 = st.number_input("rolling_mean_3", min_value=0.0, value=5300.0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Inventory & Demand")

# Inventory settings
current_inventory = st.sidebar.number_input("Current Inventory", min_value=0.0, value=5000.0)
lead_time_days = st.sidebar.number_input("Lead Time (days)", min_value=1, value=7)
demand_std_dev = st.sidebar.number_input("Demand Std Dev", min_value=0.0, value=500.0)
service_level = st.sidebar.slider("Service Level", min_value=0.8, max_value=0.99, value=0.95)

# Prepare input data
input_data = {
    "Store": store,
    "Promo": promo,
    "Promo2": promo2,
    "SchoolHoliday": school_holiday,
    "CompetitionDistance": competition_distance,
    "CompetitionOpenSince": competition_open_since,
    "Promo2OpenSince": promo2_open_since,
    "year": year,
    "month": month,
    "day": day,
    "dayOfWeek": day_of_week,
    "lag_1": lag_1,
    "rolling_mean_3": rolling_mean_3,
    "StateHoliday": state_holiday,
    "StoreType": store_type,
    "Assortment": assortment,
    "PromoInterval": promo_interval
}

# Animated divider
st.markdown('<hr class="animated-divider">', unsafe_allow_html=True)

# Enhanced prediction button
if st.button("🔍 Predict & Recommend", key="predict_btn"):
    with st.spinner("🤖 AI agents are analyzing your data..."):
        try:
            response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
            response.raise_for_status()
            result = response.json()
            predicted_demand = result.get("predicted_demand", None)
            
            # Enhanced prediction display
            st.markdown(f"""
            <div class="metric-display">
                <h3 style="color: #000000; margin-bottom: 10px;">📈 Demand Forecast</h3>
                <div class="prediction-value">{predicted_demand:,.0f} units</div>
            </div>
            """, unsafe_allow_html=True)

            # Enhanced chart - removed the wrapper div
            trend_vals = [lag_1, rolling_mean_3, predicted_demand]
            trend_labels = ["Yesterday", "3-Day Avg", "Predicted"]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('none')
            fig.patch.set_alpha(0)
            
            # Enhanced styling with black text for chart
            ax.plot(trend_labels, trend_vals, marker='o', linewidth=4, markersize=10, 
                   color='#60a5fa', markerfacecolor='#3b82f6', markeredgewidth=2, markeredgecolor='white')
            ax.fill_between(trend_labels, trend_vals, alpha=0.3, color='#60a5fa')
            
            ax.set_title("Demand Trend Analysis", fontsize=18, fontweight='bold', pad=20, color='black')
            ax.set_ylabel("Units Sold", fontsize=14, fontweight='500', color='black')
            ax.grid(True, linestyle='--', alpha=0.3, color='black')
            ax.set_facecolor('none')
            ax.tick_params(colors='black')
            
            # Enhanced annotations
            for i, (label, val) in enumerate(zip(trend_labels, trend_vals)):
                ax.annotate(f'{val:,.0f}', 
                          xy=(i, val), 
                          xytext=(0, 15), 
                          textcoords='offset points',
                          ha='center',
                          fontweight='bold',
                          fontsize=12,
                          color='white',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#3b82f6', alpha=0.8, edgecolor='white'))
            
            plt.tight_layout()
            st.pyplot(fig)

            # Enhanced recommendations
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="recommendation-card">
                    <div class="recommendation-title">🎯 Promotion Strategy</div>
                """, unsafe_allow_html=True)
                
                promo_rec = recommend_promotion(input_data, predicted_demand)
                st.markdown(f'<div class="recommendation-content">{promo_rec}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="recommendation-card">
                    <div class="recommendation-title">📦 Restocking Alert</div>
                """, unsafe_allow_html=True)
                
                restock_qty = calculate_restocking_quantity(
                    forecasted_demand=predicted_demand,
                    current_inventory=current_inventory,
                    lead_time_days=lead_time_days,
                    demand_std_dev=demand_std_dev,
                    service_level=service_level
                )
                st.markdown(f'<div class="recommendation-content">Order <strong>{restock_qty:,.0f} units</strong> to maintain {service_level:.0%} service level</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="recommendation-card">
                    <div class="recommendation-title">🔄 Inventory Optimization</div>
                """, unsafe_allow_html=True)
                
                inventory_order = dynamic_inventory_order(
                    input_data=input_data,
                    current_inventory=current_inventory,
                    lead_time_days=lead_time_days
                )
                st.markdown(f'<div class="recommendation-content">Optimal order quantity: <strong>{inventory_order:,.0f} units</strong></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Enhanced metrics
            st.markdown("### 📊 Performance Metrics")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Current Stock", f"{current_inventory:,.0f}", delta=None)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                stock_days = current_inventory / predicted_demand if predicted_demand > 0 else 0
                st.metric("Days of Stock", f"{stock_days:.1f}", delta=None)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                turnover = predicted_demand * 365 / current_inventory if current_inventory > 0 else 0
                st.metric("Inventory Turnover", f"{turnover:.1f}x", delta=None)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with metric_col4:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.metric("Service Level", f"{service_level:.0%}", delta=None)
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f" API Connection Error: {e}")
            st.info(" Please ensure the backend service is running on port 5000")