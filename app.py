import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load your trained models and the reference data
@st.cache_resource
def load_assets():
    rf = joblib.load('rf_classifier_model.pkl')
    xgb = joblib.load('xgboost_price_model.pkl')
    # We use this to calculate city medians and get 'hidden' feature values
    df = pd.read_csv('ml_ready_housing_data.csv')
    return rf, xgb, df

rf_model, xgb_model, df_clean = load_assets()

# 2. Page Configuration
st.set_page_config(page_title="Real Estate Investment Advisor", layout="centered")
st.title("🏘️ Real Estate Investment Advisor")
st.write("Powered by Random Forest & XGBoost Models")

# 3. Input Lists (Alphabetical to match your Day 3 LabelEncoder)
cities = sorted(['mumbai', 'delhi', 'bangalore', 'hyderabad', 'ahmedabad', 'chennai', 'kolkata', 'surat', 'pune', 'jaipur'])
types = sorted(['Apartment', 'Independent House', 'Villa', 'Penthouse'])
transits = sorted(['High', 'Medium', 'Low'])

# 4. UI Inputs
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("Select City", cities, index=cities.index('hyderabad'))
        prop_type = st.selectbox("Property Type", types)
        transit = st.selectbox("Transit Access", transits)
    with col2:
        size = st.number_input("Size (SqFt)", value=1500, step=50)
        price = st.number_input("Current Asking Price (Lakhs)", value=65.0, step=1.0)
        age = st.number_input("Age of Property (Years)", value=2, min_value=0, max_value=50)

if st.button("Run AI Analysis 🚀", use_container_width=True):
    # --- STEP 1: PREPARE DATA ---
    # We use the first row of your data as a 'template' for hidden features (Locality, Facing, etc.)
    input_row = df_clean.iloc[0:1].copy()
    
    # Overwrite template with User UI inputs
    input_row['City'] = cities.index(city)
    input_row['Property_Type'] = types.index(prop_type)
    input_row['Public_Transport_Accessibility'] = transits.index(transit)
    input_row['Size_in_SqFt'] = size
    input_row['Price_in_Lakhs'] = price
    input_row['Age_of_Property'] = age
    input_row['BHK'] = max(1, size // 600)
    input_row['Price_per_SqFt'] = (price * 100000) / size

    # --- STEP 2: PREDICTIONS ---
    # Get feature names from models to ensure exact column alignment
    rf_features = rf_model.feature_names_in_
    xgb_features = xgb_model.get_booster().feature_names
    
    is_good = rf_model.predict(input_row[rf_features])[0]
    future_val = xgb_model.predict(input_row[xgb_features])[0]

    # --- STEP 3: DISPLAY RESULTS ---
    st.write("---")
    if is_good == 1:
        st.success("#### ✅ APPROVED: This is a Good Investment")
    else:
        st.error("#### ❌ REJECTED: Sub-Optimal Investment")
        
    c1, c2 = st.columns(2)
    c1.metric("5-Year Price Forecast", f"₹ {future_val:.2f} L")
    profit = future_val - price
    c2.metric("Estimated Profit", f"₹ {profit:.2f} L", delta=f"{(profit/price)*100:.1f}% ROI")

    # --- STEP 4: THE EXPLAINER (Day 3 Logic) ---
    st.subheader("Why this decision?")
    city_encoded = cities.index(city)
    city_median = df_clean[df_clean['City'] == city_encoded]['Price_per_SqFt'].median()
    current_psf = (price * 100000) / size
    
    # Check against the rules you defined in day3_features.py
    is_undervalued = current_psf < city_median
    is_new = age <= 15
    has_transit = transit in ['High', 'Medium']

    col_a, col_b, col_c = st.columns(3)
    col_a.write(f"**Price Check**\n{'✅ Below' if is_undervalued else '❌ Above'} City Avg")
    col_b.write(f"**Age Check**\n{'✅ New' if is_new else '❌ Old'} (<15yrs)")
    col_c.write(f"**Transit Check**\n{'✅ Good' if has_transit else '❌ Poor'} Access")

    if not is_good:
        st.info(f"💡 Tip: For an approval in {city.title()}, the Price per SqFt should be below ₹{city_median:.0f}. Yours is ₹{current_psf:.0f}.")
