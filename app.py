import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load the trained model
with open("churn_model.sav", "rb") as f:
    model = pickle.load(f)

# Set up Streamlit UI
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🔮 Customer Churn Prediction App")
st.markdown("Enter customer details below to predict if they are likely to churn.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
multiplelines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=845.5)

# Create input dataframe
input_dict = {
    'gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phoneservice,
    'MultipleLines': multiplelines,
    'InternetService': internet,
    'OnlineSecurity': onlinesecurity,
    'OnlineBackup': onlinebackup,
    'DeviceProtection': deviceprotection,
    'TechSupport': techsupport,
    'StreamingTV': streamingtv,
    'StreamingMovies': streamingmovies,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly,
    'TotalCharges': total
}

input_df = pd.DataFrame([input_dict])

# Encode categorical variables like in training
for col in input_df.columns:
    if input_df[col].dtype == 'object':
        input_df[col], _ = pd.factorize(input_df[col])

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ The customer IS likely to churn.")
    else:
        st.success("✅ The customer is NOT likely to churn.")
