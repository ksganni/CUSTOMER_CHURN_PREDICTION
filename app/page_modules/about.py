# About page

import streamlit as st
import base64
from pathlib import Path

def show_page():
    # About This Platform section
    def get_base64_image(image_path):
        """Converting image to base64 string for HTML embedding"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception as e:
            st.error(f"Error loading image: {e}")
        return None

    # Getting the image 
    img_base64 = get_base64_image("app/assets/example.png")

    if img_base64:
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
            <div style="flex: 1; padding-right: 2rem;">
                <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: auto; border-radius: 8px;">
            </div>
            <div style="flex: 1; color: #34495e; text-align: justify;">
                <h3 style="color: #2c3e50; font-weight: 600;">
                    Customer Churn Prediction:
                </h3>
                <p style="font-size: 1.1rem; line-height: 1.6;">
                    <strong>Customer churn prediction</strong> is the process of analyzing customer data 
                    to anticipate which individuals are likely to discontinue using a company's 
                    products or services. Accurately predicting churn enables businesses to take proactive 
                    measures to retain at-risk customers, reducing customer loss and supporting long-term growth.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0; text-align: justify;">
            <h3 style="color: #2c3e50; font-weight: 600; margin-bottom: 1.5rem;">
                How This Application Works..
            </h3>
            <p style="color: #34495e; font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.2rem;">
                This application predicts whether a telecom customer is likely to churn based on their personal details, 
                subscribed services, and billing information. By analyzing the customer’s profile, 
                it provides an instant churn probability along with a risk level: Low, Medium, or High.
            </p>
            <p style="color: #34495e; font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.2rem;">
                Once you enter the customer’s details and click <strong>"Predict Churn"</strong>, the app calculates the churn likelihood using a trained 
                machine learning model. It also explains the prediction by showing which features most influenced the result, using easy-to-understand 
                charts and tables. This helps businesses quickly identify at-risk customers and take action to improve retention.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)

    # Key Platform Features header
    st.markdown("""
    <h3 style="color: #2c3e50; margin-top: 0; ">
       📝 Key Sections
    </h3>
    """, unsafe_allow_html=True)

    # 📊 Dataset Viewer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e8f4fd 0%, #bde4f4 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h3 style="color: #2980b9; margin-top: 0;">📊 Dataset</h3>
        <p style="color: #34495e; font-size: 1.1rem; line-height: 1.6;">
            Explore comprehensive customer information and understand churn trends and patterns 
            through interactive visualizations and statistical analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 🤖 Models 
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e0f2f1 0%, #a7d8d3 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h3 style="color: #00695c; margin-top: 0;">🤖 Models</h3>
        <p style="color: #34495e; font-size: 1.1rem; line-height: 1.6;">
            Compare different machine learning models and review their performance metrics 
            to identify which algorithms provide the most accurate predictions for your data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 🎯 Predictor 
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0e8ff 0%, #d1c4e9 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h3 style="color: #8e44ad; margin-top: 0;">🎯 Predictor</h3>
        <p style="color: #34495e; font-size: 1.1rem; line-height: 1.6;">
            Enter customer details to receive instant churn predictions with detailed 
            explanations and actionable insights for customer retention strategies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)

    # Sample Scenarios to Try
    st.markdown("""
    <h3 style="color: #2c3e50; margin-top: 0; ">
       💡 Sample Scenarios To Check Churn Risk
    </h3>
    """, unsafe_allow_html=True)

    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ffebee 0%, #ffeaec 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h4 style="color: #c0392b; margin-bottom: 0.5rem;">❗ High Risk: Short-Term Customer Overpaying</h4>
        <ul>
            <li>Gender: Female</li>
            <li>Senior Citizen: Yes</li>
            <li>Partner: No</li>
            <li>Dependents: No</li>
            <li>Tenure: 5 months</li>
            <li>Phone Service: Yes</li>
            <li>Multiple Lines: Yes</li>
            <li>Internet Service: Fiber Optic</li>
            <li>Online Security: No</li>
            <li>Online Backup: No</li>
            <li>Device Protection: No</li>
            <li>Tech Support: No</li>
            <li>Streaming TV: No</li>
            <li>Streaming Movies: No</li>
            <li>Contract: Month-to-Month</li>
            <li>Paperless Billing: No</li>
            <li>Payment Method: Mailed Check</li>
            <li>Monthly Charges: $95.00</li>
            <li>Total Charges: $475.00</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #fffde7 0%, #fffae0 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h4 style="color: #f39c12; margin-bottom: 0.5rem;">⚠️ Medium Risk: Moderate Tenure with Some Stability</h4>
        <ul>
            <li>Gender: Female</li>
            <li>Senior Citizen: No</li>
            <li>Partner: Yes</li>
            <li>Dependents: No</li>
            <li>Tenure: 15 months</li>
            <li>Phone Service: Yes</li>
            <li>Multiple Lines: Yes</li>
            <li>Internet Service: Fiber Optic</li>
            <li>Online Security: No</li>
            <li>Online Backup: Yes</li>
            <li>Device Protection: No</li>
            <li>Tech Support: No</li>
            <li>Streaming TV: Yes</li>
            <li>Streaming Movies: No</li>
            <li>Contract: Month-to-Month</li>
            <li>Paperless Billing: Yes</li>
            <li>Payment Method: Credit Card(Automatic)</li>
            <li>Monthly Charges: $78.90</li>
            <li>Total Charges: $1,183.50</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #e8f5e8 0%, #e5f2e5 100%); padding: 2rem; border-radius: 10px; margin: 2rem 0;">
        <h4 style="color: #27ae60; margin-bottom: 0.5rem;">✅ Low Risk: Premium Family Customer</h4>
        <ul>
            <li>Gender: Male</li>
            <li>Senior Citizen: No</li>
            <li>Partner: Yes</li>
            <li>Dependents: Yes</li>
            <li>Tenure: 65 months</li>
            <li>Phone Service: Yes</li>
            <li>Multiple Lines: Yes</li>
            <li>Internet Service: Fiber Optic</li>
            <li>Online Security: Yes</li>
            <li>Online Backup: Yes</li>
            <li>Device Protection: Yes</li>
            <li>Tech Support: Yes</li>
            <li>Streaming TV: Yes</li>
            <li>Streaming Movies: Yes</li>
            <li>Contract: Two Year</li>
            <li>Paperless Billing: Yes</li>
            <li>Payment Method: Bank Transfer(Automatic)</li>
            <li>Monthly Charges: $95.20</li>
            <li>Total Charges: $6188.00</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    