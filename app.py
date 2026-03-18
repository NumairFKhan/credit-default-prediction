import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Credit Default Predictor",
    page_icon="📊",
    layout="centered"
)

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

st.title("📊 Credit Default Predictor")
st.markdown(
    "Estimate default risk from a small set of applicant details using a trained machine learning model."
)

st.info(
    "This is an educational demo built from a public Kaggle dataset. "
    "It is not financial advice or a production underwriting tool."
)

with st.form("prediction_form"):
    st.subheader("Personal profile")

    col1, col2 = st.columns(2)
    with col1:
        age_years = st.slider("Age", min_value=18, max_value=80, value=40)
        gender = st.selectbox(
            "Gender",
            ["F", "M"],
            format_func=lambda x: "Female" if x == "F" else "Male"
        )
        family_status = st.selectbox(
            "Family status",
            ["Married", "Single / not married"]
        )

    with col2:
        years_employed = st.slider("Years employed", min_value=0, max_value=40, value=5)
        education_level = st.selectbox(
            "Education level",
            ["Higher education", "Secondary / secondary special"]
        )
        owns_car = st.selectbox(
            "Owns a car?",
            ["N", "Y"],
            format_func=lambda x: "No" if x == "N" else "Yes"
        )

    st.subheader("Financial details")

    col3, col4 = st.columns(2)
    with col3:
        annual_income = st.number_input(
            "Annual income",
            min_value=0.0,
            value=150000.0,
            step=1000.0,
            help="Total annual income of the applicant."
        )
        loan_amount = st.number_input(
            "Loan amount",
            min_value=0.0,
            value=500000.0,
            step=1000.0,
            help="Total credit amount requested."
        )
        annual_repayment = st.number_input(
            "Annual repayment amount",
            min_value=0.0,
            value=25000.0,
            step=500.0,
            help="Approximate annual annuity / repayment obligation."
        )

    with col4:
        purchase_price = st.number_input(
            "Goods / purchase price",
            min_value=0.0,
            value=450000.0,
            step=1000.0
        )
        income_type = st.selectbox(
            "Income type",
            ["Working", "Pensioner"]
        )
        loan_type = st.selectbox(
            "Loan type",
            ["Cash loans", "Revolving loans"]
        )

    st.subheader("Credit indicators")

    col5, col6 = st.columns(2)
    with col5:
        regional_risk = st.selectbox(
            "Regional risk rating",
            [1, 2, 3],
            index=1,
            help="Higher values may indicate higher regional risk."
        )
        external_score_1 = st.number_input(
            "External credit score 1",
            min_value=0.0, max_value=1.0, value=0.50, step=0.01
        )
        external_score_2 = st.number_input(
            "External credit score 2",
            min_value=0.0, max_value=1.0, value=0.50, step=0.01
        )

    with col6:
        external_score_3 = st.number_input(
            "External credit score 3",
            min_value=0.0, max_value=1.0, value=0.50, step=0.01
        )
        submitted_key_document = st.selectbox(
            "Submitted key document?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        has_work_phone = st.selectbox(
            "Has work phone?",
            [0, 1],
            index=1,
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        different_live_city = st.selectbox(
            "Lives in different city from registration?",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    submitted = st.form_submit_button("Predict default risk")

if submitted:
    input_dict = {col: np.nan for col in expected_columns}

    input_dict["AMT_INCOME_TOTAL"] = annual_income
    input_dict["AMT_CREDIT"] = loan_amount
    input_dict["AMT_ANNUITY"] = annual_repayment
    input_dict["AMT_GOODS_PRICE"] = purchase_price
    input_dict["DAYS_BIRTH"] = -age_years * 365
    input_dict["DAYS_EMPLOYED"] = -years_employed * 365
    input_dict["REGION_RATING_CLIENT_W_CITY"] = regional_risk
    input_dict["EXT_SOURCE_1"] = external_score_1
    input_dict["EXT_SOURCE_2"] = external_score_2
    input_dict["EXT_SOURCE_3"] = external_score_3
    input_dict["FLAG_DOCUMENT_3"] = submitted_key_document
    input_dict["FLAG_EMP_PHONE"] = has_work_phone
    input_dict["REG_CITY_NOT_LIVE_CITY"] = different_live_city
    input_dict["CODE_GENDER"] = gender
    input_dict["FLAG_OWN_CAR"] = owns_car
    input_dict["NAME_INCOME_TYPE"] = income_type
    input_dict["NAME_EDUCATION_TYPE"] = education_level
    input_dict["NAME_CONTRACT_TYPE"] = loan_type
    input_dict["NAME_FAMILY_STATUS"] = family_status

    input_df = pd.DataFrame([input_dict])
    input_df = input_df[expected_columns]

    probability = model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction result")
    st.metric("Estimated default probability", f"{probability:.1%}")

    if probability < 0.10:
        st.success("Risk level: Low")
        st.caption("The model views this profile as relatively lower risk.")
    elif probability < 0.20:
        st.warning("Risk level: Medium")
        st.caption("The model views this profile as moderate risk.")
    else:
        st.error("Risk level: High")
        st.caption("The model views this profile as relatively higher risk.")

    with st.expander("Show model input data"):
        st.dataframe(input_df.T)

st.markdown("---")
st.caption("Built with Streamlit, scikit-learn, XGBoost, and a public Kaggle credit dataset.")