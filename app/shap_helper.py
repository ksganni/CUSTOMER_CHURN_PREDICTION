# Using SHAP for model explainability

import shap
import pandas as pd
import streamlit as st

def explain_prediction(model,input_df,bg_df):
    explainer=shap.Explainer(model,bg_df)
    shap_values=explainer(input_df)

    st.subheader("SHAP Explanation (Local)")
    st.pyplot(shap.plots.waterfall(shap_values[0],show=False))

    st.subheader("SHAP Summary (Global)")
    st.pyplot(shap.plots.bar(shap_values,show=False))