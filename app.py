import streamlit as st
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("💼 Employer Salary Prediction App")
st.title("project by Yuvaraja")

# User Inputs
age = st.number_input("Age", min_value=17, max_value=90)
education = st.selectbox("Education", encoders['education'].classes_)
occupation = st.selectbox("Occupation", encoders['occupation'].classes_)
hours = st.slider("Hours per week", 1, 99)
gender = st.selectbox("Gender", encoders['gender'].classes_)
marital_status = st.selectbox("Marital Status", encoders['marital-status'].classes_)

# Encode input
X_input = np.array([[ 
    age,
    encoders['education'].transform([education])[0],
    encoders['occupation'].transform([occupation])[0],
    hours,
    encoders['gender'].transform([gender])[0],
    encoders['marital-status'].transform([marital_status])[0],
    0, 0  # placeholders for capital-gain and capital-loss
]])

# Predict
pred = model.predict(X_input)[0]
st.subheader("📊 Predicted Salary Class:")
st.success(">50K" if pred == 1 else "≤50K")
