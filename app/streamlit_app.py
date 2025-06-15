# Creating a Streamlit app to input customer data, Predict churn using saved model and Display SHAP interpretation

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new
from app.shap_helper import explain_prediction

# Function for classifying the churn risk
def get_risk_explanation(probability):
    if probability >= 0.7:
        return "❗High risk - Customer is most probably leaving."
    elif probability >= 0.4:
        return " ⚠️ Medium risk - Customer might leave, monitor closely."
    else:
        return " ✅ Low risk - Customer is most probably staying."

# Loading the model
try:
    with open("models/best_model.pkl", "rb") as f:
        model, reference_columns = pickle.load(f)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Loading and processing the data
try:
    df = load_data()
    df_encoded = encode_and_new(df, reference_columns=None)  # Used only for Background SHAP

    # Keeping only model-used columns and ensuring numeric dtype
    df_encoded = df_encoded[reference_columns]

    # Robust Conversion of Data Type
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            try:
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
            except:
                try:
                    df_encoded[col] = df_encoded[col].astype('category').cat.codes
                except:
                    df_encoded[col] = 0

    df_encoded = df_encoded.fillna(0)
    df_encoded = df_encoded.astype('float64')

except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

# Building the input form
st.title("**CUSTOMER CHURN PREDICTOR**")
st.markdown("Enter the Customer Information to predict churn:")

# Dynamic form in a single vertical column (No two columns)
user_input = {}
input_cols = [col for col in df.columns if col != "Churn"]

try:
    for col in input_cols:
        if df[col].dtype == "object":
            user_input[col] = st.selectbox(col, df[col].unique())
        else:
            min_value = max(0.0, float(df[col].min()))
            user_input[col] = st.number_input(col, min_value=min_value, value=float(df[col].mean()))
except KeyError as e:
    st.error(f"Missing Expected column: {e}")
    st.stop()

# Converting to DataFrame
user_df = pd.DataFrame([user_input])

# Encoding user input using training-time reference columns
try:
    user_df_encoded = encode_and_new(user_df, reference_columns=reference_columns)
    user_df_encoded = user_df_encoded[reference_columns]

    # Robust Conversion of Data Type for User Input
    for col in user_df_encoded.columns:
        if user_df_encoded[col].dtype == 'object':
            try:
                user_df_encoded[col] = pd.to_numeric(user_df_encoded[col], errors='coerce')
            except:
                try:
                    user_df_encoded[col] = user_df_encoded[col].astype('category').cat.codes
                except:
                    user_df_encoded[col] = 0

    user_df_encoded = user_df_encoded.fillna(0)
    user_df_encoded = user_df_encoded.astype('float64')

except Exception as e:
    st.error(f"Error encoding user input: {e}")
    st.stop()

# Validating the Inputs
def validate_inputs(data):
    for key, value in data.items():
        if value == '' or (isinstance(value, float) and value < 0):
            return False, f"Invalid input for {key}.Please check your entry."
    return True, "All Inputs are Valid."

# Predicting
if st.button("Predict Churn"):
    is_valid, validation_message = validate_inputs(user_input)

    if not is_valid:
        st.error(validation_message)
    else:
        # Showing the progress bar
        with st.spinner("Predicting...."):
            time.sleep(2)  # Simulating delay to ensure spinner is visible

            try:
                pred = model.predict(user_df_encoded)[0]
                prob = model.predict_proba(user_df_encoded)[0][1]

                risk_explanation = get_risk_explanation(prob)

            except Exception as pred_error:
                st.error(f"Prediction failed: {pred_error}")
                print(f"Prediction error details: {pred_error}")
                st.stop()

        st.success("Prediction completed successfully!")

        st.subheader("Prediction Result")
        st.markdown(f"**Predicted Churn:** {'Yes' if pred else 'No'}")
        st.markdown(f"**Churn Probability:** {prob:.2%}")
        st.markdown(f"**Risk Explanation:** {risk_explanation}")

        try:
            background = df_encoded.sample(min(100, len(df_encoded)), random_state=42)
            common_cols = list(set(user_df_encoded.columns) & set(background.columns))
            user_df_final = user_df_encoded[common_cols].copy()
            background_final = background[common_cols].copy()

            user_df_final = user_df_final.astype('float64')
            background_final = background_final.astype('float64')

            explain_prediction(model, user_df_final, background_final)

        except Exception as shap_error:
            st.warning("SHAP explanation failed, but prediction was successful!")
            st.info(f"SHAP Error: {str(shap_error)}")

            try:
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance (Alternative)")
                    feature_names = user_df_encoded.columns
                    importances = model.feature_importances_

                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    indices = np.argsort(importances)[::-1][:10]
                    ax.bar(range(len(indices)), importances[indices])
                    ax.set_xticks(range(len(indices)))
                    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                    ax.set_title("Top 10 Most Important Features")
                    ax.set_ylabel("Importance")
                    plt.tight_layout()
                    st.pyplot(fig)
            except Exception as fallback_error:
                st.info("Feature explanation not available")
                print(f"Fallback error: {fallback_error}")
