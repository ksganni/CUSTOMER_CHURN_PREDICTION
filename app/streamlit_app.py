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
from streamlit_option_menu import option_menu

# Setting the page title and layout
st.set_page_config(page_title="Customer Churn Predictor", layout="wide", page_icon="📊")

# Sidebar menu with reordered options
with st.sidebar:
    selected = option_menu(
        menu_title="Customer Churn App",
        options=["Home", "Dataset Viewer", "Models", "Predictor"],
        icons=["house", "table", "layers", "bar-chart"],
        default_index=0,
        orientation="vertical",
    )

# Function for classifying churn risk
def get_risk_explanation(probability):
    if probability >= 0.7:
        return "❗High risk - Customer is most probably leaving."
    elif probability >= 0.4:
        return "⚠️ Medium risk - Customer might leave, monitor closely."
    else:
        return "✅ Low risk - Customer is most probably staying."

# Load the model and try to load evaluation results
model_loaded = False
model_scores = None
reference_columns = None
model = None

try:
    with open("models/best_model.pkl", "rb") as f:
        loaded_data = pickle.load(f)
        
        # Handle different formats of saved model
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            # New format: (model, reference_columns)
            model, reference_columns = loaded_data
        elif isinstance(loaded_data, tuple):
            # Handle other tuple formats - extract just the model
            model = loaded_data[0] if hasattr(loaded_data[0], 'predict') else loaded_data
            reference_columns = None
            st.warning("⚠️ Model loaded but column information not found. Some features may not work properly.")
        else:
            # Old format: just the model
            model = loaded_data
            reference_columns = None
            st.warning("⚠️ Model loaded but column information not found. Some features may not work properly.")
    
    # Verify that model has predict method
    if not hasattr(model, 'predict'):
        st.error("❌ Loaded object is not a valid model (missing predict method)")
        st.stop()
    
    model_loaded = True
    
    # Try to load evaluation results if they exist
    try:
        with open("models/model_evaluation_results.pkl", "rb") as f:
            model_scores = pickle.load(f)
    except FileNotFoundError:
        model_scores = None
    
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load and process dataset
try:
    df = load_data()
    
    # Initialize df_encoded
    df_encoded = None
    
    # Only do encoding if we have reference columns
    if reference_columns is not None:
        df_encoded = encode_and_new(df, reference_columns=None)
        df_encoded = df_encoded[reference_columns]

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
    else:
        # If no reference columns, just do basic encoding
        df_encoded = encode_and_new(df, reference_columns=None)
        # Get reference columns from the encoded data
        reference_columns = df_encoded.columns.tolist()
        if 'Churn' in reference_columns:
            reference_columns.remove('Churn')

except Exception as e:
    st.error(f"Error processing data: {e}")
    st.stop()

# Home Page
if selected == "Home":
    # Display image with text overlay
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Load the image
        img = Image.open("app/assets/churn.png")
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Text to overlay
        text = "🏠 Customer Churn Prediction App"
        
        # Try to use a nice font, fallback to default if not available
        try:
            # Bigger font size for more prominence
            font_size = max(48, img_width // 15)  # Much larger dynamic font sizing
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # Try other common fonts for bigger text
                font_size = max(48, img_width // 15)
                font = ImageFont.truetype("Arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
        
        # Get text dimensions
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # Fallback for basic font
            text_width = len(text) * 10
            text_height = 20
        
        # Position text (centered horizontally, near the top)
        x = (img_width - text_width) // 2
        y = img_height // 8  # Position in upper portion of image
        
        # Add black text (big and bold)
        if font:
            draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))  # Pure black text
        else:
            # Fallback for basic font - still black
            draw.text((x, y), text, fill=(0, 0, 0, 255))
        
        # Convert PIL image to bytes for Streamlit
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Display the image
        st.image(img_buffer, use_container_width=True)
        
    except Exception as e:
        st.warning("Image not found or could not be loaded. Please check the path.")
        st.info(f"Image load error: {str(e)}")
        
        # Fallback: display title without image
        st.title("🏠 Customer Churn Prediction App")

    # Description below the image (without repeating the title)
    st.markdown("""
    ### Welcome to the Customer Churn Prediction App!
    
    This comprehensive app helps you predict whether a customer will churn based on their information.
    
    **Features:**
    - 📊 **Dataset Viewer**: Explore the raw dataset used for training
    - 🧩 **Models**: View performance metrics of different machine learning models
    - 🔍 **Predictor**: Make predictions with SHAP explanations
    
    **How to use:**
    1. Start by exploring the **Dataset Viewer** to understand the data
    2. Check the **Models** section to see accuracy metrics
    3. Use the **Predictor** to make churn predictions
    
    Navigate using the sidebar menu to get started!
    """)
    
# Dataset Viewer Page
elif selected == "Dataset Viewer":
    st.title("📂 Dataset Viewer")
    st.markdown("Explore the raw dataset used to train the model.")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        # Fix: Handle both string and numeric Churn columns
        if 'Churn' in df.columns:
            if df['Churn'].dtype == 'object':
                # If Churn is string ('Yes'/'No'), count 'Yes' values
                churn_count = (df['Churn'] == 'Yes').sum()
            else:
                # If Churn is numeric (1/0), sum the values
                churn_count = df['Churn'].sum()
            churn_rate = (churn_count / len(df)) * 100
        else:
            churn_rate = 0
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col4:
        st.metric("Data Types", df.dtypes.nunique())
    
    # Display dataset
    st.subheader("Raw Dataset")
    st.dataframe(df, use_container_width=True)
    
    # Basic statistics
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# Models Page
elif selected == "Models":
    st.title("🧩 Model Performance Overview")
    st.markdown("Compare the performance of different machine learning models tested for churn prediction.")

    # Check if we have actual evaluation results
    if model_scores is not None:
        # Use actual evaluation results
        st.success("✅ Displaying actual evaluation results from your training session")
        
        # Convert the loaded scores to DataFrame format
        model_names = list(model_scores.keys())
        roc_auc_means = [model_scores[name]['roc_auc_mean'] for name in model_names]
        roc_auc_stds = [model_scores[name]['roc_auc_std'] for name in model_names]
        
        model_performance = {
            'Model': model_names,
            'ROC-AUC Mean': roc_auc_means,
            'ROC-AUC Std': roc_auc_stds
        }
    else:
        # Use default/example values with a warning
        st.warning("⚠️ Using example evaluation results. Run model training to see actual scores.")
        model_performance = {
            'Model': [
                'Logistic Regression',
                'Decision Tree', 
                'Random Forest',
                'XGBoost',
                'CatBoost'
            ],
            'ROC-AUC Mean': [0.918437, 0.798340, 0.929505, 0.929416, 0.930626],
            'ROC-AUC Std': [0.065427, 0.065994, 0.047551, 0.061658, 0.058482]
        }
    
    performance_df = pd.DataFrame(model_performance)
    
    # Display performance table
    st.subheader("📊 Model Comparison")
    st.dataframe(performance_df.style.highlight_max(axis=0, subset=['ROC-AUC Mean']), 
                use_container_width=True)
    
    # Best model highlight
    best_model_idx = performance_df['ROC-AUC Mean'].idxmax()
    best_model = performance_df.iloc[best_model_idx]['Model']
    best_score = performance_df.iloc[best_model_idx]['ROC-AUC Mean']
    
    st.success(f"🏆 **Best Performing Model:** {best_model} with {best_score:.3f} ROC-AUC score")
    
    # Model details
    st.subheader("📋 Model Details")
    st.markdown(f"""
    **Selected Model:** {best_model}
    - **ROC-AUC Score:** {best_score:.3f}
    - **Why this model?** {best_model} achieved the highest ROC-AUC score, indicating excellent performance in distinguishing between churning and non-churning customers
    - **Cross-Validation:** All models were evaluated using 5-fold cross-validation
    - **Training Features:** {len(reference_columns) if reference_columns else 'N/A'} features
    
    **Model Performance Summary:**
    - **CatBoost:** {performance_df.iloc[4]['ROC-AUC Mean']:.3f} ± {performance_df.iloc[4]['ROC-AUC Std']:.3f}
    - **Random Forest:** {performance_df.iloc[2]['ROC-AUC Mean']:.3f} ± {performance_df.iloc[2]['ROC-AUC Std']:.3f}
    - **XGBoost:** {performance_df.iloc[3]['ROC-AUC Mean']:.3f} ± {performance_df.iloc[3]['ROC-AUC Std']:.3f}
    - **Logistic Regression:** {performance_df.iloc[0]['ROC-AUC Mean']:.3f} ± {performance_df.iloc[0]['ROC-AUC Std']:.3f}
    - **Decision Tree:** {performance_df.iloc[1]['ROC-AUC Mean']:.3f} ± {performance_df.iloc[1]['ROC-AUC Std']:.3f}
    
    **Note:** ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures the model's ability to distinguish between classes. A score of 1.0 is perfect, while 0.5 is random guessing.
    """)
    
    # Performance visualization
    try:
        import matplotlib.pyplot as plt
        
        st.subheader("📈 Model Performance Visualization")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC-AUC comparison with error bars
        ax1.bar(performance_df['Model'], performance_df['ROC-AUC Mean'], 
                yerr=performance_df['ROC-AUC Std'], capsize=5, color='skyblue', alpha=0.7)
        ax1.set_title('Model ROC-AUC Comparison (with Standard Deviation)')
        ax1.set_ylabel('ROC-AUC Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0.7, 1.0)
        
        # ROC-AUC ranking
        sorted_df = performance_df.sort_values('ROC-AUC Mean', ascending=True)
        colors = ['gold' if model == best_model else 'lightblue' for model in sorted_df['Model']]
        ax2.barh(sorted_df['Model'], sorted_df['ROC-AUC Mean'], color=colors)
        ax2.set_title('Model Ranking by ROC-AUC Score')
        ax2.set_xlabel('ROC-AUC Score')
        ax2.set_xlim(0.7, 1.0)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except ImportError:
        st.info("Install matplotlib to see performance visualizations: `pip install matplotlib`")

# Predictor Page
elif selected == "Predictor":
    st.title("🔍 Customer Churn Predictor")

    if model_loaded and model is not None:
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Model failed to load!")
        st.stop()

    st.markdown("### Please enter the Customer Information to predict churn:")

    user_input = {}
    input_cols = [col for col in df.columns if col != "Churn"]

    # Create input form in columns for better layout
    col1, col2 = st.columns(2)
    
    try:
        for i, col in enumerate(input_cols):
            with col1 if i % 2 == 0 else col2:
                if df[col].dtype == "object":
                    user_input[col] = st.selectbox(f"📋 {col}", df[col].unique())
                else:
                    min_value = max(0.0, float(df[col].min()))
                    user_input[col] = st.number_input(
                        f"🔢 {col}", 
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
        
        # Ensure we have the same columns as the model expects
        if reference_columns is not None:
            # Add missing columns with default values
            for col in reference_columns:
                if col not in user_df_encoded.columns:
                    user_df_encoded[col] = 0
            
            # Select only the columns the model expects
            user_df_encoded = user_df_encoded[reference_columns]
        
        # Convert data types
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

    def validate_inputs(data):
        for key, value in data.items():
            if value == '' or (isinstance(value, float) and value < 0):
                return False, f"Invalid input for {key}. Please check your entry."
        return True, "All inputs are valid."

    # Prediction button
    if st.button("🚀 Predict Churn", type="primary"):
        is_valid, validation_message = validate_inputs(user_input)

        if not is_valid:
            st.error(validation_message)
        else:
            with st.spinner("🔄 Analyzing customer data..."):
                time.sleep(2)

                try:
                    # Fix: Ensure model is the actual model object, not a tuple
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

            # Results display
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

            # SHAP explanation
            st.subheader("🧠 Model Explanation")
            try:
                if reference_columns is not None and df_encoded is not None and len(df_encoded) > 0:
                    background = df_encoded.sample(min(100, len(df_encoded)), random_state=42)
                    common_cols = list(set(user_df_encoded.columns) & set(background.columns))
                    user_df_final = user_df_encoded[common_cols].copy()
                    background_final = background[common_cols].copy()

                    user_df_final = user_df_final.astype('float64')
                    background_final = background_final.astype('float64')

                    # Use the actual model for SHAP
                    explain_prediction(actual_model, user_df_final, background_final)
                else:
                    st.info("SHAP explanation not available - missing reference data")

            except Exception as shap_error:
                st.warning("⚠️ SHAP explanation failed, but prediction was successful!")
                st.info(f"SHAP Error: {str(shap_error)}")

                # Fallback to feature importance
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