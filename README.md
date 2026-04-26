# 🏘️ Real Estate Investment Advisor: AI-Powered Price Forecasting

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline developed for real estate investment analysis. It empowers buyers and financial institutions to make data-driven decisions by classifying the viability of an investment and predicting the 5-year future value of a property.

## 🚀 Key Features
* **Investment Classification:** Utilizes a **Random Forest Classifier** (99% accuracy) to flag properties as "Good Investments" based on undervalued pricing, age, and infrastructure connectivity.
* **Price Forecasting:** Employs an **XGBoost Regressor** (R² = 0.98) to predict the estimated property value after 5 years, providing users with projected profit margins.
* **Explainable AI:** Extracts feature importance to prove *why* a property is a strong investment (e.g., highlighting that 'Property Age' and 'Price per SqFt' drive 69% of the decision).
* **Interactive Dashboard:** A fully deployed **Streamlit** web application allowing users to input property details and receive real-time financial projections.
* **Experiment Tracking:** Integrated with **MLflow** to rigorously track model parameters, metrics, and versions for professional reproducibility.

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Data Processing & EDA:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
* **Model Tracking:** MLflow
* **Deployment:** Streamlit, Joblib

## 📊 Methodology
1. **Data Engineering:** Cleaned 250,000+ rows of real estate data, handled categorical encoding, and removed extreme statistical outliers using the IQR method.
2. **Feature Engineering:** Created synthetic business metrics such as `Amenities_Count` and a custom `Good_Investment` target label based on local median price thresholds.
3. **Model Training:** Split data (80/20), tuned hyperparameters, and trained both regression and classification algorithms.
4. **UI Development:** Built a responsive, user-friendly frontend that dynamically scales minor features based on city averages to reduce user friction.

## 💻 How to Run Locally
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn mlflow streamlit joblib