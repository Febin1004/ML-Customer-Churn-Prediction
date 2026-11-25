
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("Bank Customer Churn Predictor")

st.write(
    "Enter customer details to estimate the probability "
    "of bank customer churn."
)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "final_model_outputs" / "models"
    / "final_churn_model.pkl"
)

THRESHOLD_PATH = (
    BASE_DIR / "final_model_outputs" / "best_threshold.txt"
)

# --------------------------------------------------
# LOAD MODEL AND THRESHOLD
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read().strip())

    return model, threshold


try:
    model, threshold = load_model()
except FileNotFoundError:
    st.error(
        "Model files not found. Check the final_model_outputs folder."
    )
    st.stop()

# --------------------------------------------------
# CUSTOMER INPUT FORM
# --------------------------------------------------

st.subheader("Customer Information")

with st.form("customer_form"):

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        tenure = st.number_input(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=5
        )

    with col2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        num_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4]
        )

        has_crcard = st.selectbox(
            "Has Credit Card?",
            ["Yes", "No"]
        )

        is_active = st.selectbox(
            "Is Active Member?",
            ["Yes", "No"]
        )

        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=60000.0,
            step=1000.0
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_crcard == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary
    }])

    # Get churn probability
    churn_probability = model.predict_proba(input_data)[0][1]

    # Apply selected threshold
    prediction = int(churn_probability >= threshold)

    # Determine presentation risk category
    if churn_probability < 0.30:
        risk_level = "Low"
    elif churn_probability < 0.60:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    st.divider()
    st.subheader("Prediction Results")

    st.metric(
        label="Estimated Churn Probability",
        value=f"{churn_probability:.2%}"
    )

    st.progress(float(churn_probability))

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Prediction",
            "Likely to Churn" if prediction == 1
            else "Likely to Stay"
        )

    with col2:
        st.metric(
            "Risk Category",
            risk_level
        )

    st.caption(
        f"Classification threshold: {threshold:.2f}"
    )

    if prediction == 1:
        st.error(
            "This customer is classified as likely to churn. "
            "Consider reviewing suitable customer-retention actions."
        )
    else:
        st.success(
            "This customer is classified as likely to stay "
            "based on the selected classification threshold."
        )

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------

st.divider()

st.caption(
    "Disclaimer: This project uses a synthetic bank customer "
    "dataset. Predictions are model estimates for demonstration "
    "and educational purposes, not actual banking decisions."
)
