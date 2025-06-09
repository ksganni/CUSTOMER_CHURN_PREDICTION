# Creating a Streamlit app to input customer data, Predict churn using saved model and Display SHAP interpretation

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new
from app.shap_helper import explain_prediction

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
    
    # ROBUST DATA TYPE CONVERSION
    print("=== PROCESSING BACKGROUND DATA ===")
    for col in df_encoded.columns:
        print(f"Column {col}: dtype = {df_encoded[col].dtype}")
        if df_encoded[col].dtype == 'object':
            print(f"Converting {col} from object to numeric")
            # Try different conversion methods
            try:
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
            except:
                try:
                    df_encoded[col] = df_encoded[col].astype('category').cat.codes
                except:
                    df_encoded[col] = 0
    
    # Final conversion and cleanup
    df_encoded = df_encoded.fillna(0)
    df_encoded = df_encoded.astype('float64')
    
    print("Background data dtypes after processing:")
    print(df_encoded.dtypes)
    
except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

# Building the input form ( Streamlit UI )
st.title("Customer Churn Predictor")
st.markdown("Enter Customer Information to predict churn.")

# Dynamic form
user_input = {}
input_cols = [col for col in df.columns if col != "Churn"]

try:
    for col in input_cols:
        if df[col].dtype == "object":
            user_input[col] = st.selectbox(col, df[col].unique())
        else:
            user_input[col] = st.number_input(col, value=float(df[col].mean()))
except KeyError as e:
    st.error(f"Missing Expected column: {e}")
    st.stop()

# Converting to DataFrame
user_df = pd.DataFrame([user_input])

# Encoding user input using training-time reference columns
try:
    user_df_encoded = encode_and_new(user_df, reference_columns=reference_columns)
    user_df_encoded = user_df_encoded[reference_columns]
    
    # ROBUST DATA TYPE CONVERSION FOR USER INPUT
    print("=== PROCESSING USER INPUT DATA ===")
    for col in user_df_encoded.columns:
        print(f"User column {col}: dtype = {user_df_encoded[col].dtype}")
        if user_df_encoded[col].dtype == 'object':
            print(f"Converting user {col} from object to numeric")
            try:
                user_df_encoded[col] = pd.to_numeric(user_df_encoded[col], errors='coerce')
            except:
                try:
                    user_df_encoded[col] = user_df_encoded[col].astype('category').cat.codes
                except:
                    user_df_encoded[col] = 0
    
    # Final conversion and cleanup
    user_df_encoded = user_df_encoded.fillna(0)
    user_df_encoded = user_df_encoded.astype('float64')
    
    print("User data dtypes after processing:")
    print(user_df_encoded.dtypes)
    
except Exception as e:
    st.error(f"Error encoding user input: {e}")
    st.stop()

# Predicting
if st.button("Predict Churn"):
    try:
        pred = model.predict(user_df_encoded)[0]
        prob = model.predict_proba(user_df_encoded)[0][1]
        st.markdown(f"### Prediction: {'Churn' if pred else 'No Churn'} ({prob:.2%})")

        # SHAP explanation with enhanced error handling
        try:
            print("=== PREPARING SHAP BACKGROUND ===")
            background = df_encoded.sample(min(100, len(df_encoded)), random_state=42)
            
            # Ensure background has same columns as user input
            common_cols = list(set(user_df_encoded.columns) & set(background.columns))
            user_df_final = user_df_encoded[common_cols].copy()
            background_final = background[common_cols].copy()
            
            # Double-check data types
            print("Final data check before SHAP:")
            print("User data dtypes:", user_df_final.dtypes.unique())
            print("Background data dtypes:", background_final.dtypes.unique())
            
            # Ensure everything is float64
            user_df_final = user_df_final.astype('float64')
            background_final = background_final.astype('float64')
            
            explain_prediction(model, user_df_final, background_final)
            
        except Exception as shap_error:
            st.warning("SHAP explanation failed, but prediction was successful!")
            st.info(f"SHAP Error: {str(shap_error)}")
            
            # Fallback: Show feature importance if available
            try:
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance (Alternative)")
                    feature_names = user_df_encoded.columns
                    importances = model.feature_importances_
                    
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
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
            
    except Exception as pred_error:
        st.error(f"Prediction failed: {pred_error}")
        print(f"Prediction error details: {pred_error}")