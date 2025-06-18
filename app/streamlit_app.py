# Main Streamlit app
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import pickle
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new
from streamlit_option_menu import option_menu

# Import page modules
from page_modules import home, dataset, models, predictor

# Setting the page title and layout
st.set_page_config(page_title="Customer Churn Predictor", layout="wide", page_icon="📊")

# Initialize session state for navigation if not exists
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Sidebar menu with reordered options
with st.sidebar:
    # Add a unique key to prevent conflicts
    selected = option_menu(
        menu_title="Customer Churn App",
        options=["Home", "Dataset Viewer", "Models", "Customer Churn Predictor"],
        icons=["house", "table", "layers", "bar-chart"],
        default_index=0,
        orientation="vertical",
        key="main_navigation"  # Add unique key
    )
    
    # Update session state only if selection changed
    if selected != st.session_state.current_page:
        st.session_state.current_page = selected
        # Clear prediction state when navigating away from predictor
        if 'prediction_done' in st.session_state and selected != "Customer Churn Predictor":
            st.session_state.prediction_done = False

# Loading the model and try to load evaluation results
@st.cache_resource
def load_model_and_data():
    model_loaded = False
    model_scores = None
    reference_columns = None
    model = None
    
    try:
        with open("models/best_model.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        # Handling different formats of saved model
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            # New format: (model, reference_columns)
            model, reference_columns = loaded_data
        elif isinstance(loaded_data, tuple):
            # Handle other tuple formats - extract just the model
            model = loaded_data[0] if hasattr(loaded_data[0], 'predict') else loaded_data
            reference_columns = None
            st.warning("⚠ Model loaded but column information not found. Some features may not work properly.")
        else:
            # Old format: just the model
            model = loaded_data
            reference_columns = None
            st.warning("⚠ Model loaded but column information not found. Some features may not work properly.")
        
        # Verifying that model has predict method
        if not hasattr(model, 'predict'):
            st.error("❌ Loaded object is not a valid model (missing predict method)")
            st.stop()
        model_loaded = True
        
        # Trying to load evaluation results if they exist
        try:
            with open("models/model_evaluation_results.pkl", "rb") as f:
                model_scores = pickle.load(f)
        except FileNotFoundError:
            model_scores = None
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Loading and processing dataset
    try:
        df = load_data()
        # Initializing df_encoded
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
            # Getting reference columns from the encoded data
            reference_columns = df_encoded.columns.tolist()
            if 'Churn' in reference_columns:
                reference_columns.remove('Churn')
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()
    
    return model, model_loaded, model_scores, reference_columns, df, df_encoded

# Load all data once
model, model_loaded, model_scores, reference_columns, df, df_encoded = load_model_and_data()

# Use session state for routing to prevent double rendering
current_page = st.session_state.current_page

# Route to different pages based on selection
if current_page == "Home":
    home.show_page()
elif current_page == "Dataset Viewer":
    dataset.show_page(df)
elif current_page == "Models":
    models.show_page(model_scores, reference_columns)
elif current_page == "Customer Churn Predictor":
    predictor.show_page(model, model_loaded, reference_columns, df, df_encoded)