import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

os.makedirs("eda_plots" , exist_ok=True)

# Load data
df = pd.read_csv("data/final_training.csv")

# Basic structure
print("\n--- Dataset Head ---")
print(df.head())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Descriptive Stats ---")
print(df.describe())

# Count missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Drop rows with missing values (or fill them if needed)
df = df.dropna()

# Plot distributions of numeric features
numeric_cols = ['units_sold', 'inventory_level', 'prev_units_sold',
                'prev_inventory_level', 'rolling_avg_sales']

for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"eda_plots/{col}_distribution.png")
    plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_cols + ['low_inventory_flag']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_plots/correlation_heatmap.png")
plt.close()

# Normalize features for PyTorch
features = ['inventory_level', 'day_of_week', 'month',
            'prev_units_sold', 'prev_inventory_level', 'rolling_avg_sales']
target = 'units_sold'

X = df[features].values
y = df[target].values.reshape(-1, 1)

# Standardize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save processed arrays for PyTorch
os.makedirs("processed_data", exist_ok=True)
np.save("processed_data/X.npy", X_scaled)
np.save("processed_data/y.npy", y)

print("\n✅ EDA completed and processed data saved in 'processed_data/' and plots in 'eda_plots/'")