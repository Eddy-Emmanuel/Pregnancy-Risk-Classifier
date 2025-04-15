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

# Feature labels
feature_labels = [
    "MATERNAL AGE",
    "GESTATIONAL AGE",
    "PARITY",
    "WEIGHT",
    "HEIGHT",
    "BMI",
    "SYSTOLIC BP",
    "DIASTOLIC BP",
    "URINE ANALYSIS",
    "NON NEGATIVE URINE ANALYSIS"
]

st.title("Pregnancy Risk Classifier (Meta-Ensemble)")

st.markdown("Provide values for the 10 clinical features below:")

user_input = []
for feature in feature_labels:
    value = st.number_input(label=feature, value=0.0, format="%.4f")
    user_input.append(value)

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
