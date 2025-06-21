# Dataset Viewer Page

import streamlit as st

def show_page(df):
    st.title("📂 DATASET")
    st.markdown("Explore the raw dataset used to train the model.")
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        if 'Churn' in df.columns:
            if df['Churn'].dtype == 'object':
                churn_count = (df['Churn'] == 'Yes').sum()
            else:
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

    # Add space or a divider before download section
    st.write("")  # Adds a small space
    st.write("")  # Add more lines if more space needed
    st.divider()  # Optional: adds a horizontal line

    # Download section
    st.markdown(
        "<p style='color:black; font-weight:bold;'>Download the complete dataset as a CSV file for external analysis.</p>",
        unsafe_allow_html=True
    )
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv_data,
        file_name="dataset.csv",
        mime="text/csv",
        help="Download the complete dataset as a CSV file"
    )
