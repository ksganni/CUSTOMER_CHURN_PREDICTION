# Dataset Viewer Page

import streamlit as st

def show_page(df):
    st.title("📂 Dataset Viewer")
    st.markdown("Explore the raw dataset used to train the model.")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        # Fix: Handling both string and numeric Churn columns
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
    
    # Displaying the dataset
    st.subheader("Raw Dataset")
    st.dataframe(df, use_container_width=True)
    
    # Basic statistics
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)