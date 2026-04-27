import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Labmentix Real Estate AI", layout="wide")
st.title("🏘️ Real Estate Investment Advisor")

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    rf = joblib.load('rf_classifier_model.pkl')
    xgb = joblib.load('xgboost_price_model.pkl')
    df = pd.read_csv('ml_ready_housing_data.csv')
    features = joblib.load('model_features.pkl')
    return rf, xgb, df, features

rf_model, xgb_model, df_clean, model_features = load_assets()

# --- SIDEBAR ---
cities = sorted(['mumbai', 'delhi', 'bangalore', 'hyderabad', 'ahmedabad', 'chennai', 'kolkata', 'surat', 'pune', 'jaipur'])
types = sorted(['Apartment', 'Independent House', 'Villa', 'Penthouse'])
transits = ['High', 'Medium', 'Low']

st.sidebar.header("Property Details")
city = st.sidebar.selectbox("City", options=cities, index=3)
property_type = st.sidebar.selectbox("Property Type", options=types)
transport = st.sidebar.selectbox("Transit Access", options=transits)
size_sqft = st.sidebar.slider("Size (SqFt)", 500, 5000, 1500)
current_price = st.sidebar.number_input("Current Asking Price (Lakhs)", 10.0, 1000.0, 65.0)
age = st.sidebar.slider("Age of Property (Years)", 0, 50, 2)

# --- PROCESSING ---
if st.button("Analyze Property 🚀", type="primary"):
    # 1. Map inputs to index values (0, 1, 2...)
    city_idx = cities.index(city)
    type_idx = types.index(property_type)
    transit_idx = transits.index(transport)

    # 2. Use a real row from your data as a template to avoid missing columns
    input_row = df_clean[df_clean['City'] == city_idx].iloc[0:1].copy()
    
    # 3. Overwrite with UI values
    input_row['City'] = city_idx
    input_row['Property_Type'] = type_idx
    input_row['Public_Transport_Accessibility'] = transit_idx
    input_row['Size_in_SqFt'] = size_sqft
    input_row['Price_in_Lakhs'] = current_price
    input_row['Age_of_Property'] = age
    input_row['BHK'] = max(1, size_sqft // 600)
    input_row['Price_per_SqFt'] = (current_price * 100000) / size_sqft 

    # 4. STRICT ALIGNMENT: Reorder columns to match 'model_features.pkl' exactly
    X_input = input_row[model_features].astype(float)

    # --- PREDICTIONS ---
    # Random Forest expects all features including Price_per_SqFt
    is_good = rf_model.predict(X_input)[0]
    
    # XGBoost usually drops Price_per_SqFt (Day 4 Logic)
    xgb_features = [f for f in model_features if f != 'Price_per_SqFt']
    future_price = xgb_model.predict(X_input[xgb_features])[0]

    # --- RESULTS ---
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if is_good == 1:
            st.success("#### ✅ APPROVED: High-Yield Investment")
            st.balloons()
        else:
            st.error("#### ❌ REJECTED: Sub-Optimal Investment")
            # Logic explainer
            city_median = df_clean[df_clean['City'] == city_idx]['Price_per_SqFt'].median()
            user_psf = (current_price * 100000) / size_sqft
            if user_psf > city_median:
                st.warning(f"Price Tip: ₹{user_psf:.0f}/sqft is above the {city} average of ₹{city_median:.0f}.")

    with col2:
        st.metric("5-Year Forecast", f"₹ {future_price:.2f} L")
        profit = future_price - current_price
        st.metric("Estimated Profit", f"₹{profit:.2f} L", delta=f"{(profit/current_price)*100:.1f}% ROI")
