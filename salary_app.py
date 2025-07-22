import streamlit as st
import pandas as pd
import joblib

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="SmartSalary Predictor",
    page_icon="💼",
    layout="centered"
)

# --- LOAD MODEL & FEATURES ---
model = joblib.load("salary_predictor_reduced.pkl")
input_features = joblib.load("input_features.pkl")

# --- PREDEFINED MAPPINGS ---
workclass_options = {
    0: "Private", 1: "Self-emp-not-inc", 2: "Self-emp-inc", 3: "Federal-gov",
    4: "Local-gov", 5: "State-gov", 6: "Without-pay", 7: "Never-worked"
}

marital_status_options = {
    0: "Married-civ-spouse", 1: "Divorced", 2: "Never-married",
    3: "Separated", 4: "Widowed", 5: "Married-spouse-absent", 6: "Other"
}

occupation_options = {
    0: "Tech-support", 1: "Craft-repair", 2: "Other-service", 3: "Sales",
    4: "Exec-managerial", 5: "Prof-specialty", 6: "Handlers-cleaners",
    7: "Machine-op-inspct", 8: "Adm-clerical", 9: "Farming-fishing",
    10: "Transport-moving", 11: "Priv-house-serv", 12: "Protective-serv",
    13: "Armed-Forces", 14: "Unknown"
}

# ✅ UPDATED: Realistic Native Country Options
native_country_options = {
    0: "United States", 1: "Cambodia", 2: "England", 3: "Puerto Rico", 4: "Canada",
    5: "Germany", 6: "Outlying US (Guam-USVI-etc)", 7: "India", 8: "Japan", 9: "Greece",
    10: "South", 11: "China", 12: "Cuba", 13: "Iran", 14: "Honduras", 15: "Philippines",
    16: "Italy", 17: "Poland", 18: "Jamaica", 19: "Vietnam", 20: "Mexico", 21: "Portugal",
    22: "Ireland", 23: "France", 24: "Dominican Republic", 25: "Laos", 26: "Ecuador",
    27: "Taiwan", 28: "Haiti", 29: "Colombia", 30: "Hungary", 31: "Guatemala",
    32: "Nicaragua", 33: "Scotland", 34: "Thailand", 35: "Yugoslavia", 36: "El Salvador",
    37: "Trinadad & Tobago", 38: "Peru", 39: "Holand-Netherlands"
}

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
        .main { background-color: #f6f9fc; padding: 20px; border-radius: 10px; }
        .stButton>button { background-color: #008080; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("💼 SmartSalary Predictor (Powered by LightGBM)")
st.markdown("_Know your worth in seconds — Machine Learning Model: **LightGBM**_")


# --- HELP SECTION ---
with st.expander("ℹ️ How It Works", expanded=False):
    st.write("""
    - This app predicts whether a person’s income exceeds 50K/year.
    - Model is trained on census data with LightGBM.
    - Encoded numeric values represent categories – editable below.
    """)

# --- INPUT FORM ---
with st.form("salary_form"):
    st.subheader("📋 Employee Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 30)
        workclass = st.selectbox("Workclass", list(workclass_options.values()))
        marital_status = st.selectbox("Marital Status", list(marital_status_options.values()))
        occupation = st.selectbox("Occupation", list(occupation_options.values()))

    with col2:
        education_num = st.slider("Education Level (years)", 1, 16, 10)
        sex = st.selectbox("Sex", ["Female", "Male"])
        hours_per_week = st.slider("Work Hours/Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", list(native_country_options.values()))

    submitted = st.form_submit_button("🔍 Predict Salary")

# --- PREDICTION ---
if submitted:
    # Reverse mapping from label to code
    workclass_code = list(workclass_options.keys())[list(workclass_options.values()).index(workclass)]
    marital_code = list(marital_status_options.keys())[list(marital_status_options.values()).index(marital_status)]
    occupation_code = list(occupation_options.keys())[list(occupation_options.values()).index(occupation)]
    native_country_code = list(native_country_options.keys())[list(native_country_options.values()).index(native_country)]
    sex_code = 0 if sex == "Female" else 1

    input_data = pd.DataFrame([[
        age, workclass_code, education_num, marital_code,
        occupation_code, sex_code, hours_per_week, native_country_code
    ]], columns=input_features)

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]
    confidence = round(max(proba) * 100, 1)
    label = ">50K" if pred == 1 else "≤50K"

    # --- OUTPUT ---
    st.markdown("---")
    if pred == 1:
        st.success(f"🤑 **Predicted Salary Class:** `{label}`")
    else:
        st.warning(f"💼 **Predicted Salary Class:** `{label}`")

    st.metric("🔎 Prediction Confidence", f"{confidence}%")
