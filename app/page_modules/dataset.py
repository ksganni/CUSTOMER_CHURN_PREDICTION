# Dataset Viewer page

import streamlit as st

def show_page(df):
    # Styled header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #e0ecff 0%, #f5f7fa 100%); border-radius: 12px; margin-bottom: 2rem;">
        <h1 style="color: #2c3e50; margin: 0; font-size: 2.5rem;">
            📂 Dataset Viewer
        </h1>
        <p style="color: #34495e; font-size: 1.2rem; margin: 0.5rem 0 0 0; opacity: 0.9;">
            Explore the raw dataset used to train the model
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Overview section
    st.markdown("### <span style='color:black; font-weight:bold;'>📊 Dataset Overview</span>", unsafe_allow_html=True)
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

    # Raw dataset
    st.markdown("### <span style='color:black; font-weight:bold;'>📋 Raw Dataset</span>", unsafe_allow_html=True)
    st.markdown("<p style='color:black; font-weight:bold;'>Complete dataset with all customer records and features used for model training.</p>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    # Dataset statistics
    st.markdown("### <span style='color:black; font-weight:bold;'>📈 Dataset Statistics</span>", unsafe_allow_html=True)
    st.markdown("<p style='color:black; font-weight:bold;'>Statistical summary of numerical features in the dataset.</p>", unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)

    # Download section
    st.markdown("### <span style='color:black; font-weight:bold;'>Download Dataset</span>", unsafe_allow_html=True)
    st.markdown("<p style='color:black; font-weight:bold;'>Download the complete dataset as a CSV file for external analysis.</p>", unsafe_allow_html=True)
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv_data,
        file_name="dataset.csv",
        mime="text/csv",
        help="Download the complete dataset as a CSV file"
    )
