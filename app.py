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
@st.cache_resource
def load_assets():
    rf_model = joblib.load('rf_classifier_model.pkl')
    xgb_model = joblib.load('xgboost_price_model.pkl')
    # Use the pre-processed data for consistency
    df_clean = pd.read_csv('ml_ready_housing_data.csv')
    return rf_model, xgb_model, df_clean

rf_model, xgb_model, df_clean = load_assets()

# --- DEFINE CATEGORICAL LISTS ---
# These must match the order/names used in your original dataset
city_options = ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'ahmedabad', 'chennai', 'kolkata', 'surat', 'pune', 'jaipur']
property_type_options = ['Apartment', 'Independent House', 'Villa', 'Penthouse']
transport_options = ['High', 'Medium', 'Low']

# --- SIDEBAR INPUTS ---
st.sidebar.header("Property Details")

# Now we use the human-readable lists for the select boxes
city = st.sidebar.selectbox("City", options=city_options)
property_type = st.sidebar.selectbox("Property Type", options=property_type_options)
transport = st.sidebar.selectbox("Public Transport Accessibility", options=transport_options)

size_sqft = st.sidebar.slider("Size (SqFt)", 500, 5000, 1500)
current_price = st.sidebar.number_input("Current Asking Price (Lakhs)", min_value=10.0, max_value=1000.0, value=150.0)
age = st.sidebar.slider("Age of Property (Years)", 0, 50, 5)

# --- INFER MINOR FEATURES ---
# We find a row in our data that matches the selected city (encoded) 
# to fill in State and Locality automatically.
city_encoded_val = city_options.index(city)
city_data = df_clean[df_clean['City'] == city_encoded_val].iloc[0]

# Construct the raw dataframe for prediction
input_data = pd.DataFrame([{
    'State': city_data['State'], # Already encoded in ml_ready_data
    'City': city,
    'Locality': city_data['Locality'], # Already encoded in ml_ready_data
    'Property_Type': property_type,
    'BHK': max(1, size_sqft // 600),
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
# We manually map the categorical inputs to numbers to match the model training
input_data['Property_Type'] = property_type_options.index(property_type)
input_data['Public_Transport_Accessibility'] = transport_options.index(transport)
input_data['City'] = city_options.index(city)

# Map simple binary defaults (1 for Yes/Ready/Builder/Semi-furnished etc.)
# This ensures the model receives numbers, not strings.
binary_cols = ['Furnished_Status', 'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status']
for col in binary_cols:
    input_data[col] = 1 

# Convert everything to float/int to avoid the pandas error
input_data = input_data.astype(float)

# Drop Price_per_SqFt for XGBoost as per Day 4 logic
xgb_input = input_data.drop(columns=['Price_per_SqFt'], errors='ignore')

# --- PREDICTIONS & UI ---
st.write("---")
st.subheader("AI Investment Analysis")

col1, col2 = st.columns(2)

if st.button("Analyze Property", type="primary"):
    # 1. Classification
    is_good = rf_model.predict(input_data)[0]
    
    # 2. Regression
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
