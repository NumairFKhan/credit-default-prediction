import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Default Predictor", layout="centered")

MODEL_PATH = "credit_default_xgb_pipeline.joblib"
COLUMNS_PATH = "expected_columns.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_expected_columns():
    return joblib.load(COLUMNS_PATH)

model = load_model()
expected_columns = load_expected_columns()

st.title("Credit Default Predictor")
st.markdown("Enter applicant details and generate a default-risk prediction from the trained XGBoost pipeline.")

st.subheader("Applicant inputs")

# ---- Numeric inputs ----
amt_income_total = st.number_input("AMT_INCOME_TOTAL", min_value=0.0, value=150000.0, step=1000.0)
amt_credit = st.number_input("AMT_CREDIT", min_value=0.0, value=500000.0, step=1000.0)
amt_annuity = st.number_input("AMT_ANNUITY", min_value=0.0, value=25000.0, step=500.0)
amt_goods_price = st.number_input("AMT_GOODS_PRICE", min_value=0.0, value=450000.0, step=1000.0)

days_birth = st.number_input("DAYS_BIRTH (negative number)", value=-15000, step=100)
days_employed = st.number_input("DAYS_EMPLOYED (negative number; use 365243 if unknown)", value=-2000, step=100)

region_rating_client_w_city = st.number_input("REGION_RATING_CLIENT_W_CITY", min_value=1, max_value=3, value=2, step=1)
ext_source_1 = st.number_input("EXT_SOURCE_1", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
ext_source_2 = st.number_input("EXT_SOURCE_2", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
ext_source_3 = st.number_input("EXT_SOURCE_3", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

flag_document_3 = st.selectbox("FLAG_DOCUMENT_3", [0, 1], index=0)
flag_emp_phone = st.selectbox("FLAG_EMP_PHONE", [0, 1], index=1)
reg_city_not_live_city = st.selectbox("REG_CITY_NOT_LIVE_CITY", [0, 1], index=0)

# ---- Categorical inputs ----
code_gender = st.selectbox("CODE_GENDER", ["F", "M"])
flag_own_car = st.selectbox("FLAG_OWN_CAR", ["N", "Y"])
name_income_type = st.selectbox("NAME_INCOME_TYPE", ["Working", "Pensioner"])
name_education_type = st.selectbox(
    "NAME_EDUCATION_TYPE",
    ["Higher education", "Secondary / secondary special"]
)
name_contract_type = st.selectbox("NAME_CONTRACT_TYPE", ["Cash loans", "Revolving loans"])
name_family_status = st.selectbox("NAME_FAMILY_STATUS", ["Married", "Single / not married"])

# Start with all expected columns as NaN
input_dict = {col: np.nan for col in expected_columns}

# Overwrite only the fields we collected
input_dict["AMT_INCOME_TOTAL"] = amt_income_total
input_dict["AMT_CREDIT"] = amt_credit
input_dict["AMT_ANNUITY"] = amt_annuity
input_dict["AMT_GOODS_PRICE"] = amt_goods_price
input_dict["DAYS_BIRTH"] = days_birth
input_dict["DAYS_EMPLOYED"] = days_employed
input_dict["REGION_RATING_CLIENT_W_CITY"] = region_rating_client_w_city
input_dict["EXT_SOURCE_1"] = ext_source_1
input_dict["EXT_SOURCE_2"] = ext_source_2
input_dict["EXT_SOURCE_3"] = ext_source_3
input_dict["FLAG_DOCUMENT_3"] = flag_document_3
input_dict["FLAG_EMP_PHONE"] = flag_emp_phone
input_dict["REG_CITY_NOT_LIVE_CITY"] = reg_city_not_live_city
input_dict["CODE_GENDER"] = code_gender
input_dict["FLAG_OWN_CAR"] = flag_own_car
input_dict["NAME_INCOME_TYPE"] = name_income_type
input_dict["NAME_EDUCATION_TYPE"] = name_education_type
input_dict["NAME_CONTRACT_TYPE"] = name_contract_type
input_dict["NAME_FAMILY_STATUS"] = name_family_status

input_df = pd.DataFrame([input_dict])

# Force correct column order
input_df = input_df[expected_columns]

if st.button("Predict default risk"):
    probability = model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction")
    st.write(f"Estimated default probability: **{probability:.1%}**")

    if probability < 0.10:
        st.success("Risk level: Low")
    elif probability < 0.20:
        st.warning("Risk level: Medium")
    else:
        st.error("Risk level: High")

    st.subheader("Input preview")
    st.dataframe(input_df.T)