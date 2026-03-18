import streamlit as st

st.set_page_config(page_title="Credit Default Predictor", layout="centered")

st.title("Credit Default Predictor")
st.markdown("Enter a few applicant details to estimate default risk.")

st.subheader("Applicant inputs")

income = st.number_input("Annual income", min_value=0.0, value=150000.0, step=1000.0)
credit_amount = st.number_input("Credit amount", min_value=0.0, value=500000.0, step=1000.0)
goods_price = st.number_input("Goods price", min_value=0.0, value=450000.0, step=1000.0)
age_years = st.slider("Age", min_value=18, max_value=80, value=40)
years_employed = st.slider("Years employed", min_value=0, max_value=40, value=5)
has_car = st.selectbox("Owns a car?", ["No", "Yes"])
gender = st.selectbox("Gender", ["Female", "Male"])
education = st.selectbox(
    "Education level",
    ["Secondary / secondary special", "Higher education"]
)

if st.button("Predict default risk"):
    risk_score = 0.0

    if income < 100000:
        risk_score += 0.20
    if credit_amount > 700000:
        risk_score += 0.20
    if goods_price > 600000:
        risk_score += 0.10
    if age_years < 30:
        risk_score += 0.15
    if years_employed < 2:
        risk_score += 0.20
    if has_car == "No":
        risk_score += 0.05
    if education == "Secondary / secondary special":
        risk_score += 0.10
    if gender == "Male":
        risk_score += 0.02

    probability = min(risk_score, 0.95)

    st.subheader("Prediction")
    st.write(f"Estimated default probability: **{probability:.1%}**")

    if probability < 0.15:
        st.success("Risk level: Low")
    elif probability < 0.35:
        st.warning("Risk level: Medium")
    else:
        st.error("Risk level: High")

    st.caption("This is a demo scoring app. It is not yet connected to your trained Kaggle model.")