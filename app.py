import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ===================== LOAD MODEL =====================
pipeline = joblib.load(r'C:\Users\user\Desktop\project\credit risk assisment\pipeline.pkl')
# Extract parts
preprocessor = pipeline.named_steps['preprocessor']
model = pipeline.named_steps['model']

st.title("💳 Credit Risk Prediction + Explainability")

st.write("Fill customer details:")

# ===================== INPUT =====================
CODE_GENDER = st.selectbox("Gender", ["M", "F"])
FLAG_OWN_CAR = st.selectbox("Own Car", ["Y", "N"])
FLAG_OWN_REALTY = st.selectbox("Own House", ["Y", "N"])
FLAG_PHONE = st.selectbox("Has Phone", ["Y", "N"])
FLAG_EMAIL = st.selectbox("Has Email", ["Y", "N"])
FLAG_WORK_PHONE = st.selectbox("Has Work Phone", ["Y", "N"])
CNT_FAM_MEMBERS = st.number_input("Family Members", 1, 10, 1)

CNT_CHILDREN = st.number_input("Children", 0, 10, 0)
AMT_INCOME_TOTAL = st.number_input("Income", 0.0)

NAME_INCOME_TYPE = st.selectbox("Income Type", [
    "Working", "Commercial associate", "Pensioner", "State servant"
])

NAME_EDUCATION_TYPE = st.selectbox("Education", [
    "Secondary / secondary special", "Higher education"
])

NAME_FAMILY_STATUS = st.selectbox("Family Status", [
    "Married", "Single / not married"
])

NAME_HOUSING_TYPE = st.selectbox("Housing", [
    "House / apartment", "With parents"
])

OCCUPATION_TYPE = st.selectbox("Occupation", [
    "Laborers", "Core staff", "Managers", "Drivers", "Unknown"
])

EMPLOYED_YEARS = st.number_input("Employment Years", 0.0)
AGE = st.number_input("Age", 18.0)

PAST_DELAY_COUNT = st.number_input("Past Delay Count", 0)
PAST_MONTH_COUNT = st.number_input("Past Month Count", 0)

# ===================== PREDICT =====================
if st.button("Predict"):

    input_df = pd.DataFrame([{
        'CODE_GENDER': CODE_GENDER,
        'FLAG_OWN_CAR': FLAG_OWN_CAR,
        'FLAG_OWN_REALTY': FLAG_OWN_REALTY,
        'CNT_CHILDREN': CNT_CHILDREN,
        'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
        'FLAG_PHONE': 1 if FLAG_PHONE == "Y" else 0,
        'FLAG_EMAIL': 1 if FLAG_EMAIL == "Y" else 0,
        'FLAG_WORK_PHONE': 1 if FLAG_WORK_PHONE == "Y" else 0,
        'CNT_FAM_MEMBERS': CNT_FAM_MEMBERS,
        'NAME_INCOME_TYPE': NAME_INCOME_TYPE,
        'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
        'NAME_FAMILY_STATUS': NAME_FAMILY_STATUS,
        'NAME_HOUSING_TYPE': NAME_HOUSING_TYPE,
        'OCCUPATION_TYPE': OCCUPATION_TYPE,
        'EMPLOYED_YEARS': EMPLOYED_YEARS,
        'AGE': AGE,
        'PAST_DELAY_COUNT': PAST_DELAY_COUNT,
        'PAST_MONTH_COUNT': PAST_MONTH_COUNT
    }])

    # ===================== PREDICTION =====================
    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ High Risk\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Low Risk\nProbability: {prob:.2f}")

    