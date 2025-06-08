import shap
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def explain_prediction(model, user_df, background_df):
    """
    Generate SHAP explanations with comprehensive error handling
    """
    try:
        print("=== SHAP EXPLANATION START ===")
        print(f"User data shape: {user_df.shape}")
        print(f"Background data shape: {background_df.shape}")
        print(f"User data dtypes: {user_df.dtypes.unique()}")
        print(f"Background data dtypes: {background_df.dtypes.unique()}")
        
        # Ensure all data is float64
        user_clean = user_df.astype('float64')
        bg_clean = background_df.astype('float64')
        
        # Check for any remaining non-numeric data
        if not all(user_clean.dtypes == 'float64'):
            raise ValueError("User data contains non-numeric columns")
        if not all(bg_clean.dtypes == 'float64'):
            raise ValueError("Background data contains non-numeric columns")
        
        print("Creating SHAP explainer...")
        explainer = shap.Explainer(model, bg_clean)
        
        print("Computing SHAP values...")
        shap_values = explainer(user_clean)
        
        print("Creating SHAP visualization...")
        st.subheader("🔍 Prediction Explanation")
        
        # Handle multi-output models (binary classifiers)
        print(f"SHAP values shape: {shap_values.values.shape}")
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if we have multi-output (binary classifier)
        if len(shap_values.values.shape) > 2:
            # Multi-output model - use the positive class (index 1)
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            shap_vals_for_df = shap_values.values[0, :, 1]
        elif shap_values.values.shape[1] > 1:
            # Binary classifier with 2 outputs - use positive class
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            shap_vals_for_df = shap_values.values[0, :, 1]
        else:
            # Single output
            shap.plots.waterfall(shap_values[0], show=False)
            shap_vals_for_df = shap_values.values[0]
        
        st.pyplot(fig)
        plt.close()
        
        # Show feature contributions
        st.subheader("📊 Feature Contributions")
        shap_df = pd.DataFrame({
            'Feature': user_clean.columns,
            'Value': user_clean.iloc[0].values,
            'SHAP_Value': shap_vals_for_df
        })
        shap_df['Abs_SHAP'] = abs(shap_df['SHAP_Value'])
        shap_df = shap_df.sort_values('Abs_SHAP', ascending=False)
        
        st.dataframe(shap_df.head(10))
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"SHAP Error: {error_msg}")
        
        # More specific error handling
        if "dtype" in error_msg.lower() or "cast" in error_msg.lower():
            st.error("Data type error in SHAP explanation")
            st.info("The model prediction was successful, but explanation requires all numeric data.")
        else:
            st.error(f"SHAP explanation failed: {error_msg}")
        
        return False

def safe_shap_fallback(model, user_df):
    """
    Simple fallback when SHAP fails
    """
    try:
        if hasattr(model, 'feature_importances_'):
            st.subheader("📈 Model Feature Importance")
            
            importance_df = pd.DataFrame({
                'Feature': user_df.columns,
                'Importance': model.feature_importances_,
                'User_Value': user_df.iloc[0].values
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = importance_df.head(10)
            ax.barh(range(len(top_features)), top_features['Importance'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 10 Most Important Features')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.dataframe(importance_df)
            
    except Exception as e:
        st.info("Feature explanation not available")
        print(f"Fallback explanation error: {e}")