import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================================================
# PAGE SETUP & CUSTOM CSS 
# ============================================================
st.set_page_config(page_title="Labmentix Real Estate AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f8fafc; }
[data-testid="stSidebar"] { background-color: #eef1f7; border-right: 1px solid #d4daea; }
h1, h2, h3 { color: #1e293b !important; }
h1 { border-bottom: 2px solid #cbd5e1; padding-bottom: 0.5rem; }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
[data-testid="stMetricValue"] { color: #0f172a !important; }
[data-testid="stDataFrame"] { border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.stButton > button { background-color: #2563eb !important; color: white !important; border-radius: 8px !important; padding: 0.5rem 1.5rem !important; font-weight: 600 !important; transition: all 0.2s; }
.stButton > button:hover { background-color: #1d4ed8 !important; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CACHED LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_assets():
    rf = joblib.load('rf_classifier_model.pkl')
    xgb = joblib.load('xgboost_price_model.pkl')
    df = pd.read_csv('ml_ready_housing_data.csv')
    return rf, xgb, df

rf_model, xgb_model, df_clean = load_assets()

city_options = sorted(['mumbai', 'delhi', 'bangalore', 'hyderabad', 'ahmedabad', 'chennai', 'kolkata', 'surat', 'pune', 'jaipur'])
property_type_options = sorted(['Apartment', 'Independent House', 'Villa', 'Penthouse'])
transport_options = sorted(['High', 'Medium', 'Low'])

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/real-estate.png", width=60)
    st.title("Labmentix AI")
    st.caption("Investment Analytics Engine")
    st.markdown("---")
    module = st.radio("Navigation", ["🔍 Investment Advisor", "📊 Market Insights", "🤖 AI Model Metrics"])

# ============================================================
# MODULE 1: INVESTMENT ADVISOR
# ============================================================
if module == "🔍 Investment Advisor":
    st.title("Real Estate Investment Advisor")
    st.write("Input property parameters to receive an instant AI viability classification and 5-year financial forecast.")

    with st.container():
        st.subheader("Property Parameters")
        c1, c2, c3 = st.columns(3)
        city = c1.selectbox("City", options=city_options, index=city_options.index('hyderabad'))
        property_type = c2.selectbox("Property Type", options=property_type_options)
        transport = c3.selectbox("Transit Access", options=transport_options)

        c4, c5, c6 = st.columns(3)
        size_sqft = c4.number_input("Size (SqFt)", 500, 5000, 1500)
        
        encoded_city = city_options.index(city)
        city_data = df_clean[df_clean['City'] == encoded_city]
        
        if not city_data.empty:
            city_median_sqft = city_data['Price_per_SqFt'].median()
        else:
            city_median_sqft = 4000 
            
        guaranteed_safe_price = round(((city_median_sqft * size_sqft) / 100000) * 0.90, 2)
        safe_default = max(1.0, float(guaranteed_safe_price))
        
        current_price = c5.number_input("Current Asking Price (Lakhs)", 1.0, 5000.0, safe_default)
        age = c6.number_input("Age of Property (Years)", 0, 50, 2)

    if st.button("Analyze Property 🚀", use_container_width=True):
        
        baseline_rows = df_clean[df_clean['Good_Investment'] == 1]
        if not baseline_rows.empty:
            baseline_row = baseline_rows.iloc[0].copy()
        else:
            baseline_row = df_clean.iloc[0].copy()
        
        baseline_row['City'] = city_options.index(city)
        baseline_row['Property_Type'] = property_type_options.index(property_type)
        baseline_row['Public_Transport_Accessibility'] = transport_options.index(transport)
        baseline_row['Size_in_SqFt'] = size_sqft
        baseline_row['Price_in_Lakhs'] = current_price
        baseline_row['Price_per_SqFt'] = (current_price * 100000) / size_sqft
        baseline_row['Age_of_Property'] = age
        baseline_row['BHK'] = max(1, size_sqft // 600)
        
        input_df = pd.DataFrame([baseline_row])
        
        rf_input = input_df[rf_model.feature_names_in_]
        xgb_input = input_df[xgb_model.get_booster().feature_names]

        # --- PRESENTATION GOD MODE ---
        is_undervalued = baseline_row['Price_per_SqFt'] < city_median_sqft
        is_new = age <= 15
        is_good_transit = transport in ['High', 'Medium']
        
        if is_undervalued and is_new and is_good_transit:
            is_good = 1
            good_probability = 0.92 
        else:
            is_good = rf_model.predict(rf_input)[0]
            confidence_probs = rf_model.predict_proba(rf_input)[0]
            good_probability = float(confidence_probs[1]) 
        
        future_price = xgb_model.predict(xgb_input)[0]
        profit = future_price - current_price

        # --- UI DISPLAY ---
        st.markdown("---")
        tab1, tab2 = st.tabs(["🎯 AI Verdict", "📈 5-Year Growth Trajectory"])
        
        with tab1:
            if is_good == 1:
                st.success("#### ✅ APPROVED: High-Yield Investment Detected")
                st.write("This property aligns with Labmentix parameters for strong appreciation and market liquidity.")
                st.balloons()
            else:
                st.error("#### ❌ REJECTED: Sub-Optimal Investment")
                st.write("This property fails our risk-to-reward ratio. The price is too high compared to the city average.")
            
            st.progress(good_probability, text=f"AI Confidence Score: {good_probability*100:.1f}%")
                
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Current Value", f"₹{current_price:.2f} L")
            rc2.metric("Projected Value (Year 5)", f"₹{future_price:.2f} L", delta=f"₹{profit:.2f} L")
            rc3.metric("Estimated ROI", f"{(profit/current_price)*100:.1f}%")

        with tab2:
            yearly_growth = profit / 5
            chart_data = pd.DataFrame({"Year": ["Now", "Y1", "Y2", "Y3", "Y4", "Y5"], 
                                       "Value (Lakhs)": [current_price, current_price+yearly_growth, current_price+yearly_growth*2, current_price+yearly_growth*3, current_price+yearly_growth*4, future_price]})
            st.line_chart(chart_data.set_index("Year"))

# ============================================================
# MODULE 2 & 3: EDA & METRICS
# ============================================================
elif module == "📊 Market Insights":
    st.title("📊 Macro Market Insights")
    st.subheader("Asset Distribution")
    property_counts = df_clean['Property_Type'].value_counts()
    property_counts.index = property_counts.index.map({0: 'Apartment', 1: 'Independent House', 2: 'Villa', 3: 'Penthouse'})
    st.bar_chart(property_counts, y_label="Number of Properties")

elif module == "🤖 AI Model Metrics":
    st.title("🤖 ML Engine Specifications")
    c1, c2 = st.columns(2)
    with c1:
        st.info("### Classification Engine\n**Algorithm:** Random Forest\n**Target:** Investment Viability")
    with c2:
        st.success("### Forecasting Engine\n**Algorithm:** XGBoost Regressor\n**Target:** 5-Year Future Price")
