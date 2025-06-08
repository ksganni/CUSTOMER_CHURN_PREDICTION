import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new
from app.shap_helper import explain_prediction, safe_shap_fallback

# Loading saved model and reference columns
try:
    with open("models/best_model.pkl", "rb") as f:
        model, reference_columns = pickle.load(f)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Loading and preprocessing full dataset for SHAP background
try:
    df = load_data()
    df_encoded = encode_and_new(df, reference_columns=None)
    df_encoded = df_encoded[reference_columns]

    # Converting to float64 safely
    df_encoded = df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')
except Exception as e:
    st.error(f"❌ Error preparing background data: {e}")
    st.stop()

# App title and form
st.title("📉 Customer Churn Prediction")
st.markdown("Enter customer details below to predict churn:")

user_input = {}
input_cols = [col for col in df.columns if col != "Churn"]

try:
    for col in input_cols:
        if df[col].dtype == "object":
            user_input[col] = st.selectbox(col, df[col].unique())
        else:
            user_input[col] = st.number_input(col, value=float(df[col].mean()))
except KeyError as e:
    st.error(f"❌ Missing Expected column: {e}")
    st.stop()

user_df = pd.DataFrame([user_input])

# Encoding user input
try:
    user_df_encoded = encode_and_new(user_df, reference_columns=reference_columns)
    user_df_encoded = user_df_encoded[reference_columns]
    user_df_encoded = user_df_encoded.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float64')
except Exception as e:
    st.error(f"❌ Error encoding user input: {e}")
    st.stop()

# Prediction and explanation
if st.button("🔍 Predict Churn"):
    try:
        prediction = model.predict(user_df_encoded)[0]
        probability = model.predict_proba(user_df_encoded)[0][1]
        st.markdown(f"### Prediction: **{'Churn' if prediction else 'No Churn'}** ({probability:.2%} probability)")

        # Preparing SHAP background sample
        background = df_encoded.sample(n=min(100, len(df_encoded)), random_state=42)
        common_cols = list(set(user_df_encoded.columns) & set(background.columns))
        user_final = user_df_encoded[common_cols].copy()
        background_final = background[common_cols].copy()

        user_final = user_final.astype('float64')
        background_final = background_final.astype('float64')

        success = explain_prediction(model, user_final, background_final)

        if not success:
            safe_shap_fallback(model, user_df_encoded)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        print(f"Prediction error: {e}")
