# 🚔 Crime Predictor Dashboard

A **Streamlit-based web application** that predicts **crime locations (district/police station)** and **case closure probabilities** using historical **First Information Report (FIR)** data.  
This tool leverages **machine learning models** such as **RandomForest, XGBoost, and Neural Networks**, and provides deep insights into similar historical cases when lodging new FIRs.

---

## 🧭 Overview

The **Crime Predictor Dashboard** allows users to:

- 📂 Upload FIR datasets (Excel format)
- 🧠 Predict **crime-prone locations** using trained ML models
- 🔍 Analyze **similar past cases** and their outcomes
- ⚖️ Predict **case closure probabilities** across four outcomes:
  - Police Station
  - Court
  - Self-Compromise
  - Withdraw

It includes **interactive data visualizations**, **robust error handling**, and **logging support** for a smooth analytical experience.

---

## ⚙️ Features

✅ **Data Upload** — Upload Excel files (e.g., `FIR_data_2022.xlsx`) directly via sidebar  
✅ **Location Prediction** — Multi-model prediction using RandomForest, XGBoost, and Neural Networks  
✅ **Similar Case Analysis** — Displays top 5 similar historical FIRs with their closure status  
✅ **Disposition Prediction** — Neural network model for closure outcome probabilities  
✅ **Visual Analytics** — Confusion matrices, treemaps, and feature importance plots  
✅ **Model Persistence** — Automatically saves models, encoders, and scalers for reuse  
✅ **Error Logging** — Tracks all errors and activities in `crime_predictor.log`  

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python (scikit-learn, XGBoost, TensorFlow) |
| **Visualization** | Plotly |
| **Data Handling** | Pandas, NumPy |
| **Similarity Matching** | FuzzyWuzzy |
| **Persistence** | Joblib |
| **Logging** | Python Logging Library |

---

## 🛠️ Installation

### **Prerequisites**
- Python **3.8+**
- Pip (Python package manager)

### **Install Dependencies**
```bash
pip install streamlit pandas numpy scikit-learn tensorflow scikeras plotly fuzzywuzzy python-Levenshtein joblib xgboost
