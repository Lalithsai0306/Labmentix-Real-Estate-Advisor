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

if st.button("Analyze Property 🚀", type="primary", use_container_width=True):
    # 1. Classification & Regression
    is_good = rf_model.predict(input_data)[0]
    future_price = xgb_model.predict(xgb_input)[0]
    profit = future_price - current_price
    
    # Create Professional Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Verdict", "📈 Financial Forecast", "🤖 AI Insights"])
    
    with tab1:
        st.subheader("Investment Decision")
        if is_good == 1:
            st.success("✅ **APPROVED:** The AI classifies this as a High-Yield Investment.")
            
         else:
            st.error("❌ **REJECTED:** This property does not meet high-yield criteria. High risk of low appreciation.")
            
    with tab2:
        st.subheader("5-Year Value Projection")
        col1, col2 = st.columns(2)
        col1.metric(label="Current Asking Price", value=f"₹ {current_price:.2f} L")
        col2.metric(label="Estimated 5-Year Value", value=f"₹ {future_price:.2f} L", delta=f"₹ {profit:.2f} L Profit")
        
        # Create dynamic yearly data for a chart
        yearly_growth = (future_price - current_price) / 5
        chart_data = pd.DataFrame({
            "Year": ["Current", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"],
            "Estimated Value (Lakhs)": [current_price, current_price + yearly_growth, current_price + yearly_growth*2, 
                                        current_price + yearly_growth*3, current_price + yearly_growth*4, future_price]
        })
        st.line_chart(chart_data.set_index("Year"))
        
    with tab3:
        st.subheader("How did the AI decide?")
        with st.expander("View Model Logic"):
            st.write("The **Random Forest Classifier** evaluated this property based on:")
            st.write(f"- **Valuation:** Compared your price of ₹{current_price}L against the city average.")
            st.write(f"- **Property Age:** Factored in that the property is {age} years old.")
            st.write(f"- **Transport:** Adjusted score based on '{transport}' transit accessibility.")
            st.caption("The XGBoost Regressor calculated the exact future price based on 21 micro-features.")
