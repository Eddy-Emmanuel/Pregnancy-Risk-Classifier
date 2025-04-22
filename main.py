import streamlit as st
import joblib
import numpy as np
from scipy import stats

# Load base and meta models
loaded_model_1 = joblib.load("model_1.pkl")
loaded_model_2 = joblib.load("model_2.pkl")
loaded_model_3 = joblib.load("model_3.pkl")
loaded_model_4 = joblib.load("model_4.pkl")
loaded_model_5 = joblib.load("model_5.pkl")
loaded_model_6 = joblib.load("model_6.pkl")
loaded_meta_model_1 = joblib.load("meta_model_1.pkl")
loaded_meta_model_2 = joblib.load("meta_model_2.pkl")

# Class labels
class_indices = {0: "High", 1: "Mid", 2: "Low"}
urine_mapper = {"0":0, "+1":1, "+2":2, "+3":3}

st.title("Safe Motherhood Predictor (SMP)")

maternal_age = st.number_input("Maternal Age (years)", min_value=15, max_value=50, step=1, value=29)

gestational_age = st.number_input("Gestational Age (weeks)", min_value=3.0, max_value=42.0, step=0.5, value=32.0)

parity = st.number_input("Parity (Number of pregnancies)", min_value=0, max_value=8, step=1, value=1)

weight = st.number_input("Weight (kg)", min_value=24.0, max_value=130.0, step=1.0, value=69.0)

height = st.number_input("Height (m)", min_value=1.0, max_value=2.2, step=0.01, value=1.65)

bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, step=0.1, value=27.3)

systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, step=1, value=120)

diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=120, step=1, value=66)

urine_analysis = urine_mapper[st.selectbox(
    "Urine Analysis (Proteinuria)",
    options=[
        '0', 
        '+1', 
        '+2', 
        '+3'
    ]
)]

non_negative_urine_analysis = urine_analysis if urine_analysis == 0 else 1

user_input = [
    maternal_age,
    gestational_age,
    parity,
    weight,
    height,
    bmi,
    systolic_bp,
    diastolic_bp,
    urine_analysis,
    non_negative_urine_analysis
]


if st.button("Classify"):
    input_array = np.expand_dims(user_input, axis=0)

    # Predict with base models
    base_preds = np.array([
        loaded_model_1.predict(input_array)[0],
        loaded_model_2.predict(input_array)[0],
        loaded_model_3.predict(input_array)[0],
        loaded_model_4.predict(input_array)[0],
        loaded_model_5.predict(input_array)[0],
        loaded_model_6.predict(input_array)[0][0]  
    ])

    meta_data = np.expand_dims(base_preds, axis=0)

 
    meta_pred_1 = loaded_meta_model_1.predict(meta_data)[0]
    meta_pred_2 = loaded_meta_model_2.predict(meta_data)[0][0]


    final_pred = stats.mode([meta_pred_1, meta_pred_2], keepdims=True).mode[0]
    final_class = class_indices[final_pred]

    st.success(f"Predicted Class: **{final_class}**")
