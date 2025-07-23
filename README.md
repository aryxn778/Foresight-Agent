# Foresight Agent

*Foresight Agent* is a Smart Supply Chain Forecasting & Decision System that uses machine learning and agentic AI to predict demand and optimize key logistics operations. Built using PySpark and Flask, it provides accurate forecasting (MAPE ≈ 11%) and intelligent decision-making agents to improve inventory flow and reduce cost.

---

## 🔧 Tech Stack

- *PySpark* (GBTRegressor for demand prediction)
- *Pandas* and *NumPy*
- *Flask* (for inference API)
- *REST APIs* (for testing and integration)
- *Agentic AI Architecture* (6 intelligent agents)
- *Requests* (for API testing)

---

## 🎯 Core Features

- 📈 *Demand Forecasting* using Gradient Boosted Trees in PySpark
- 🤖 *Agentic AI Components*:
  - Inventory Reordering Agent
  - Restocking Policy Optimizer
  - Promotion Recommender Agent
  - Supply Chain Cost Optimizer
  - Multi-Objective Decision Engine
  - Feedback Loop for Model Correction
- 🌐 *Flask Inference API* for easy local and cloud deployment
- 📊 *Test Script* to validate predictions via JSON

---

## 📂 Project Structure

Foresight-Agent/
│
├── api/                   # Flask API backend
│   ├── app.py             # Inference API
│   └── test_api.py        # Script to test predictions
│
├── model/                 # Saved PySpark model (excluded via .gitignore)
├── checkpoints/           # Checkpoints for agents (excluded)
├── data/                  # Training/testing data
├── notebooks/             # Optional notebooks
├── .gitignore
└── README.md

---

## 🚀 How to Run Locally

1. *Start the Flask API:*

```bash
cd api
python app.py

2. * In a new Terminal While keeping the previous on running*

python test_api.py

You'll get something like this
{ "predicted_demand": 5649.11 }

Model Performance
	•	Evaluation Metric: MAPE (Mean Absolute Percentage Error)
	•	Current MAPE: ~11% (on holdout test set)
	•	Model Used: PySpark GBTRegressor with advanced time-series features

IMPORTANT-: Use Python, pyspark, java versions which are compatible with each other

Thank You!
