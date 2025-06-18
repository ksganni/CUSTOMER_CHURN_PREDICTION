# Predictor Page

import streamlit as st
import pandas as pd
import time
import numpy as np
from src.feature_engineering import encode_and_new
from app.shap_helper import explain_prediction

# Function for classifying churn risk
def get_risk_explanation(probability):
    if probability >= 0.7:
        return "❗High risk - Customer is most probably leaving."
    elif probability >= 0.4:
        return "⚠ Medium risk - Customer might leave, monitor closely."
    else:
        return "✅ Low risk - Customer is most probably staying."

def validate_inputs(data):
    for key, value in data.items():
        if value == '' or (isinstance(value, float) and value < 0):
            return False, f"Invalid input for {key}. Please check your entry."
    return True, "All inputs are valid."

def show_page(model, model_loaded, reference_columns, df, df_encoded):
    st.title("🔍 CUSTOMER CHURN PREDICTOR")

    if model_loaded and model is not None:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Model failed to load!")
        st.stop()

    st.markdown("### **Please enter the Customer Information to predict churn:**")

    # Initialize session state variable for prediction
    if 'prediction_done' not in st.session_state:
        st.session_state['prediction_done'] = False

    user_input = {}
    input_cols = [col for col in df.columns if col != "Churn"]

    emoji_mapping = {
        'gender': '👥',
        'seniorcitizen': '🧓',
        'partner': '🤝',
        'dependents': '👶',
        'tenure': '⏱',
        'phoneservice': '📞',
        'multiplelines': '📱',
        'internetservice': '🌐',
        'onlinesecurity': '🔐',
        'onlinebackup': '☁',
        'deviceprotection': '🔧',
        'techsupport': '🛠',
        'streamingtv': '📺',
        'streamingmovies': '🎬',
        'contract': '📝',
        'paperlessbilling': '📧',
        'paymentmethod': '💳',
        'monthlycharges': '💵',
        'totalcharges': '💰'
    }

    try:
        for col in input_cols:
            col_emoji = emoji_mapping.get(col.lower(), '🔢')
            if df[col].dtype == "object":
                user_input[col] = st.selectbox(f"{col_emoji} {col}", df[col].unique())
            else:
                min_value = max(0.0, float(df[col].min()))
                user_input[col] = st.number_input(
                    f"{col_emoji} {col}",
                    min_value=min_value,
                    value=float(df[col].mean()),
                    help=f"Range: {df[col].min():.2f} - {df[col].max():.2f}"
                )
    except KeyError as e:
        st.error(f"Missing expected column: {e}")
        st.stop()

    user_df = pd.DataFrame([user_input])

    try:
        user_df_encoded = encode_and_new(user_df, reference_columns=reference_columns)
        if reference_columns is not None:
            for col in reference_columns:
                if col not in user_df_encoded.columns:
                    user_df_encoded[col] = 0
            user_df_encoded = user_df_encoded[reference_columns]

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

    # Button triggers prediction and saves encoded data in session state
    if st.button("🚀 Predict Churn", type="primary"):
        is_valid, validation_message = validate_inputs(user_input)
        if not is_valid:
            st.error(validation_message)
        else:
            st.session_state['prediction_done'] = True
            st.session_state['user_df_encoded'] = user_df_encoded

    # Display prediction results only if prediction_done is True
    if st.session_state.get('prediction_done', False):
        user_df_encoded = st.session_state['user_df_encoded']
        pred = None
        prob = None
        risk_explanation = None
        actual_model = None

        with st.spinner("🔄 Analyzing customer data..."):
            time.sleep(1)  # Reduced from 10 seconds to 1 second
            try:
                if isinstance(model, tuple):
                    actual_model = model[0] if hasattr(model[0], 'predict') else model
                else:
                    actual_model = model

                pred = actual_model.predict(user_df_encoded)[0]
                prob = actual_model.predict_proba(user_df_encoded)[0][1]
                risk_explanation = get_risk_explanation(prob)

            except Exception as pred_error:
                st.error(f"Prediction failed: {pred_error}")
                st.error(f"Model type: {type(model)}")
                st.error(f"Model has predict method: {hasattr(model, 'predict')}")
                st.stop()

        st.success("✅ Prediction completed successfully!")

        st.subheader("🎯 Prediction Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Churn", "Yes" if pred else "No")
        with col2:
            st.metric("Churn Probability", f"{prob:.2%}")
        with col3:
            risk_color = "🔴" if prob >= 0.7 else "🟡" if prob >= 0.4 else "🟢"
            st.metric("Risk Level", f"{risk_color}")

        st.info(f"**Risk Explanation:** {risk_explanation}")

        st.subheader("🔬 Model Explanation")
        try:
            if reference_columns is not None and df_encoded is not None and len(df_encoded) > 0:
                background = df_encoded.sample(min(100, len(df_encoded)), random_state=42)
                common_cols = list(set(user_df_encoded.columns) & set(background.columns))
                user_df_final = user_df_encoded[common_cols].copy()
                background_final = background[common_cols].copy()
                user_df_final = user_df_final.astype('float64')
                background_final = background_final.astype('float64')
                explain_prediction(actual_model, user_df_final, background_final)
            else:
                st.info("SHAP explanation not available - missing reference data")
        except Exception as shap_error:
            st.warning("⚠ SHAP explanation failed, but prediction was successful!")
            st.info(f"SHAP Error: {str(shap_error)}")
            try:
                if hasattr(actual_model, 'feature_importances_'):
                    st.subheader("📊 Feature Importance (Alternative)")
                    feature_names = user_df_encoded.columns
                    importances = actual_model.feature_importances_
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    indices = np.argsort(importances)[::-1][:10]
                    ax.bar(range(len(indices)), importances[indices], color='lightblue')
                    ax.set_xticks(range(len(indices)))
                    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
                    ax.set_title("Top 10 Most Important Features")
                    ax.set_ylabel("Importance")
                    plt.tight_layout()
                    st.pyplot(fig)
            except Exception as fallback_error:
                st.info("Feature explanation not available")