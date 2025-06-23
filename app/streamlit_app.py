# Main Streamlit 

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import pickle
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new
from page_modules import about, dataset, models, predictor

# Setting the page config
st.set_page_config(page_title="Customer Churn Predictor", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .nav-container {
        background: linear-gradient(90deg, #a7c7e7, #c7dff7);
        padding: 0.8rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        border-radius: 8px;
    }

    /* Title link - More specific selectors and !important for override */
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: white !important;
        margin: 0;
        letter-spacing: 0.5px;
        text-decoration: none !important;
        cursor: pointer;
    }

    /* Multiple selectors to ensure no hover color change */
    .app-title:hover,
    .app-title:focus,
    .app-title:active,
    .app-title:visited {
        color: white !important;
        text-decoration: none !important;
    }

    /* Additional override for any inherited link styles */
    a.app-title,
    a.app-title:hover,
    a.app-title:focus,
    a.app-title:active,
    a.app-title:visited {
        color: white !important;
        text-decoration: none !important;
    }

    .nav-right {
        display: flex;
        gap: 2rem;
        align-items: center;
    }

    /* Nav links (About, Dataset, Models) */
    .nav-link {
        color: white;
        font-size: 0.95rem;
        font-weight: 400;
        cursor: pointer;
        transition: color 0.2s ease;
        text-decoration: none !important;
    }

    .nav-link:hover {
        color: #e3f2fd !important;
        text-decoration: none !important;
    }

    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .main .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Getting the current page from query params
query_params = st.query_params
current_page = query_params.get("page", "Customer Churn Predictor")

# Top navigation bar 
# Top navigation bar 
st.markdown(f"""
<style>
.nav-container {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 1rem;
    background: linear-gradient(90deg, #6ca0d8, #9bbbe2);  /* darker than before */
    border-radius: 8px;
    margin: -1rem -1rem 2rem -1rem;
    color: #1e1e1e;
}}

.app-title {{
    font-weight: bold;
    font-size: 1.35rem;
    text-decoration: none;
    color: #222 !important;
    letter-spacing: 0.5px;
}}

.nav-right {{
    display: flex;
    gap: 2rem;
    align-items: center;
}}

.nav-link {{
    display: flex;
    align-items: center;
    font-size: 1rem;
    text-decoration: none;
    color: #2a2a2a !important;
    font-weight: 500;
    transition: color 0.2s ease;
}}

.nav-link:hover {{
    color: #ffffff !important;
}}

.nav-link svg {{
    width: 18px;
    height: 18px;
    margin-right: 6px;
    stroke: currentColor;
    stroke-width: 2;
    fill: none;
}}
</style>

<div class="nav-container">
    <a href="?page=Customer%20Churn%20Predictor" class="app-title">📊 CUSTOMER CHURN PREDICTOR</a>
    <div class="nav-right">
        <a href="?page=About" class="nav-link">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
              <circle cx="16" cy="16" r="14"/>
              <line x1="16" y1="12" x2="16" y2="22"/>
              <circle cx="16" cy="8" r="1.5"/>
            </svg>
            About
        </a>
        <a href="?page=Dataset" class="nav-link">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
                <rect x="3" y="3" width="26" height="26" rx="2" ry="2"/>
                <line x1="3" y1="11" x2="29" y2="11"/>
                <line x1="3" y1="19" x2="29" y2="19"/>
                <line x1="11" y1="3" x2="11" y2="29"/>
                <line x1="19" y1="3" x2="19" y2="29"/>
            </svg>
            Dataset
        </a>
        <a href="?page=Models" class="nav-link">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
                <polygon points="16 4 4 10 16 16 28 10 16 4"/>
                <polyline points="4 16 16 22 28 16"/>
                <polyline points="4 22 16 28 28 22"/>
            </svg>
            Models
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Loading the model and data
@st.cache_resource
def load_model_and_data():
    model_loaded = False
    model_scores = None
    reference_columns = None
    model = None

    try:
        with open("models/best_model.pkl", "rb") as f:
            loaded_data = pickle.load(f)
        if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
            model, reference_columns = loaded_data
        elif isinstance(loaded_data, tuple):
            model = loaded_data[0] if hasattr(loaded_data[0], 'predict') else loaded_data
            reference_columns = None
            st.warning("⚠ Model loaded but column information not found.")
        else:
            model = loaded_data
            reference_columns = None
            st.warning("⚠ Model loaded but column information not found.")

        if not hasattr(model, 'predict'):
            st.error("❌ Invalid model: missing predict method")
            st.stop()
        model_loaded = True

        try:
            with open("models/model_evaluation_results.pkl", "rb") as f:
                model_scores = pickle.load(f)
        except FileNotFoundError:
            model_scores = None

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    try:
        df = load_data()
        df_encoded = None
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
            df_encoded = encode_and_new(df, reference_columns=None)
            reference_columns = df_encoded.columns.tolist()
            if 'Churn' in reference_columns:
                reference_columns.remove('Churn')
    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

    return model, model_loaded, model_scores, reference_columns, df, df_encoded

# Loading once
model, model_loaded, model_scores, reference_columns, df, df_encoded = load_model_and_data()

# Page routing
if current_page == "Customer Churn Predictor":
    predictor.show_page(model, model_loaded, reference_columns, df, df_encoded)
elif current_page == "About":
    about.show_page()
elif current_page == "Models":
    models.show_page(model_scores, reference_columns)
elif current_page == "Dataset":
    dataset.show_page(df)
else:
    st.error("Page not found.")
