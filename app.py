import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- PAGE SETUP ---
st.set_page_config(page_title="Labmentix Real Estate AI", layout="wide")
st.title("🏘️ Real Estate Investment Advisor")
st.markdown("Developed for Labmentix | Powered by XGBoost & Random Forest")

# --- LOAD DATA & MODELS ---
@st.cache_resource # This speeds up the app by loading models only once
def load_assets():
    rf_model = joblib.load('rf_classifier_model.pkl')
    xgb_model = joblib.load('xgboost_price_model.pkl')
    # Load the clean data to fit our LabelEncoders on the fly
    df_clean = pd.read_csv('cleaned_india_housing_prices.csv')
    return rf_model, xgb_model, df_clean

rf_model, xgb_model, df_clean = load_assets()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Property Details")

# We use the unique values from our dataset for the dropdowns
city = st.sidebar.selectbox("City", df_clean['City'].unique())
property_type = st.sidebar.selectbox("Property Type", df_clean['Property_Type'].unique())
transport = st.sidebar.selectbox("Public Transport Accessibility", ['High', 'Medium', 'Low'])

size_sqft = st.sidebar.slider("Size (SqFt)", 500, 5000, 1500)
current_price = st.sidebar.number_input("Current Asking Price (Lakhs)", min_value=10.0, max_value=1000.0, value=150.0)
age = st.sidebar.slider("Age of Property (Years)", 0, 50, 5)

# --- INFER MINOR FEATURES ---
# To make the app user-friendly, we don't ask the user for all 20 columns. 
# We assume standard values for the minor features based on the chosen city.
city_data = df_clean[df_clean['City'] == city].iloc[0]

# Construct the raw dataframe for prediction
input_data = pd.DataFrame([{
    'State': city_data['State'],
    'City': city,
    'Locality': city_data['Locality'],
    'Property_Type': property_type,
    'BHK': max(1, size_sqft // 600), # Rough estimate
    'Size_in_SqFt': size_sqft,
    'Price_in_Lakhs': current_price,
    'Price_per_SqFt': current_price / size_sqft,
    'Furnished_Status': 'Semi-furnished',
    'Floor_No': 3,
    'Total_Floors': 10,
    'Age_of_Property': age,
    'Nearby_Schools': 5,
    'Nearby_Hospitals': 3,
    'Public_Transport_Accessibility': transport,
    'Parking_Space': 'Yes',
    'Security': 'Yes',
    'Facing': 'East',
    'Owner_Type': 'Builder',
    'Availability_Status': 'Ready_to_Move',
    'Amenities_Count': 4
}])

# --- ENCODING ---
# We must apply the exact same LabelEncoding as Day 3
categorical_cols = ['Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility', 
                    'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status', 
                    'State', 'City', 'Locality']

le = LabelEncoder()
for col in categorical_cols:
    le.fit(df_clean[col].astype(str)) # Learn the vocabulary
    # Safely transform the input
    input_data[col] = input_data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)

# Drop Price_per_SqFt for XGBoost (as per Day 4 logic)
xgb_input = input_data.drop(columns=['Price_per_SqFt'], errors='ignore')

# --- PREDICTIONS & UI ---
st.write("---")
st.subheader("AI Investment Analysis")

col1, col2 = st.columns(2)

if st.button("Analyze Property", type="primary"):
    # 1. Classification (Good/Bad Investment)
    is_good = rf_model.predict(input_data)[0]
    
    # 2. Regression (Future Price)
    future_price = xgb_model.predict(xgb_input)[0]
    
    with col1:
        if is_good == 1:
            st.success("✅ **APPROVED:** This is classified as a Good Investment.")
        else:
            st.error("❌ **REJECTED:** This property does not meet high-yield investment criteria.")
            
    with col2:
        st.info(f"📈 **5-Year Forecast:** ₹ {future_price:.2f} Lakhs")
        profit = future_price - current_price
        st.metric(label="Estimated Profit", value=f"₹ {profit:.2f} Lakhs", delta=f"{(profit/current_price)*100:.1f}%")

st.write("---")
st.caption("Note: This tool uses XGBoost and Random Forest algorithms trained on historical data. Market conditions may vary.")