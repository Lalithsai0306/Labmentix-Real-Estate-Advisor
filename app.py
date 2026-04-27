import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load your trained models and the reference data
@st.cache_resource
def load_simple_assets():
    rf = joblib.load('rf_classifier_model.pkl')
    xgb = joblib.load('xgboost_price_model.pkl')
    df = pd.read_csv('ml_ready_housing_data.csv')
    return rf, xgb, df

rf_model, xgb_model, df_clean = load_simple_assets()

# 2. Simple UI
st.title("Simple Real Estate Advisor")
st.write("This version uses the exact logic from your Day 3-5 scripts.")

# 3. Inputs (Matching your Day 3 Encodings)
# Note: These lists must be ALPHABETICAL to match the LabelEncoder from Day 3
cities = sorted(['mumbai', 'delhi', 'bangalore', 'hyderabad', 'ahmedabad', 'chennai', 'kolkata', 'surat', 'pune', 'jaipur'])
types = sorted(['Apartment', 'Independent House', 'Villa', 'Penthouse'])
transits = sorted(['High', 'Medium', 'Low'])

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("Select City", cities, index=3) # Default Hyderabad
    prop_type = st.selectbox("Property Type", types)
    transit = st.selectbox("Transit Access", transits)
with col2:
    size = st.number_input("Size (SqFt)", value=1500)
    price = st.number_input("Price (Lakhs)", value=45.0)
    age = st.number_input("Age (Years)", value=5)

if st.button("Run AI Analysis"):
    # Create the input row exactly as the model expects
    # We use a median 'template' from your data for the hidden columns (Locality, etc.)
    input_row = df_clean.iloc[0:1].copy()
    
    # Update with your UI inputs
    input_row['City'] = cities.index(city)
    input_row['Property_Type'] = types.index(prop_type)
    input_row['Public_Transport_Accessibility'] = transits.index(transit)
    input_row['Size_in_SqFt'] = size
    input_row['Price_in_Lakhs'] = price
    input_row['Age_of_Property'] = age
    input_row['BHK'] = max(1, size // 600)
    
    # Required for the Day 3 'Good_Investment' logic check
    input_row['Price_per_SqFt'] = (price * 100000) / size

    # Align columns for each model
    rf_features = rf_model.feature_names_in_
    xgb_features = xgb_model.get_booster().feature_names
    
    # PREDICT
    is_good = rf_model.predict(input_row[rf_features])[0]
    future_val = xgb_model.predict(input_row[xgb_features])[0]
    
    # DISPLAY RESULTS
    st.write("---")
    if is_good == 1:
        st.success("✅ APPROVED: Good Investment")
    else:
        st.error("❌ REJECTED: Sub-Optimal Investment")
        
    st.metric("5-Year Forecast", f"₹ {future_val:.2f} Lakhs")
    
    # --- WHY IS IT REJECTING? (The Explainer) ---
    st.subheader("Data Check (Why the AI decided this):")
    city_encoded = cities.index(city)
    city_median = df_clean[df_clean['City'] == city_encoded]['Price_per_SqFt'].median()
    current_psf = (price * 100000) / size
    
    st.write(f"1. Your Price per SqFt: **{current_psf:.0f}**")
    st.write(f"2. {city.title()} Median Price per SqFt: **{city_median:.0f}**")
    
    if current_psf > city_median:
        st.warning(f"⚠️ REASON: Your price is higher than the city average. AI thinks it's overpriced.")
    if age > 15:
        st.warning("⚠️ REASON: The property is older than 15 years.")
    if transit == 'Low':
        st.warning("⚠️ REASON: Poor transit access reduces investment value.")
