import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Bank Customer Churn Predictor")
st.markdown("Enter customer details below to predict if they are at risk of leaving the bank.")

# Load the trained model
@st.cache_resource
def load_model():
    if os.path.exists('churn_model.pkl'):
        return joblib.load('churn_model.pkl')
    return None

model = load_model()

if model is None:
    st.error("Model not found. Please run `train_model.py` first to generate `churn_model.pkl`.")
else:
    # Create the layout for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)

    with col2:
        balance = st.number_input("Balance ($)", min_value=0.0, value=50000.0)
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_crcard = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        salary = st.number_input("Estimated Salary ($)", min_value=0.0, value=60000.0)

    # Convert selectbox inputs to binary where necessary
    has_crcard_bin = 1 if has_crcard == "Yes" else 0
    is_active_bin = 1 if is_active == "Yes" else 0

    # Submit button
    if st.button("Predict Churn Risk", type="primary", use_container_width=True):
        # Create input dataframe matching the training schema
        input_data = pd.DataFrame([[
            credit_score, geography, gender, age, tenure, 
            balance, num_products, has_crcard_bin, is_active_bin, salary
        ]], columns=[
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1]

        # Display results
        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 **High Risk of Churn!** (Probability: {prediction_proba:.2%})")
            st.markdown("This customer exhibits patterns strongly associated with account closure. Recommend immediate retention strategies.")
        else:
            st.success(f"✅ **Safe Customer.** (Churn Probability: {prediction_proba:.2%})")
            st.markdown("This customer is likely to remain with the bank.")
