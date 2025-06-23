# Dataset Viewer Page

import streamlit as st

def show_page(df):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h3 style="color: #2c3e50;font-weight: 600;">
            📂 About The Dataset
        </h3>
        <p style="color: #34495e; font-size: 1.1rem; line-height: 1.6;">
            This dataset contains detailed information about telecom customers, including their service subscriptions, 
            account details, and demographics. It also indicates whether each customer has churned. The data is useful 
            for analyzing customer behavior and identifying patterns that lead to churn.
        </p>        
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)

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

    st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)
    
    # Displaying the dataset
    st.subheader("Raw Dataset")
    st.dataframe(df, use_container_width=True)

    st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)
    
    # Basic statistics
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)

    # Download section
    st.markdown(
        "<p style='color:black; font-weight:bold;'>Download the complete dataset as a CSV file.</p>",
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
