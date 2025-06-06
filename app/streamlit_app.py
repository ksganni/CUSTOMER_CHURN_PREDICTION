# Creating a Streamlit app to input customer data, Predict churn using saved model and Display SHAP interpretation

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import streamlit as st
import pandas as pd
import pickle
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new
from app.shap_helper import explain_prediction

# Loading the model
with open("models/best_model.pkl","rb") as f:
    model=pickle.load(f)

# Loading and processing the data
df=load_data()
df_encoded=encode_and_new(df)

# Building the input form
st.title("Customer Churn Predictor")
st.markdown("Enter Customer Information to predict churn.")

# Dynamic form
user_input={}
input_cols=[col for col in df.columns if col!="Churn"]

try:
    for col in input_cols:
        if df[col].dtype=="object":
            user_input[col]=st.selectbox(col,df[col].unique())
        else:
            user_input[col]=st.number_input(col,value=float(df[col].mean()))

except KeyError as e:
    st.error(f"Missing Expected column: {e}")
    st.stop()

# Converting to DataFrame
user_df=pd.DataFrame([user_input])
user_df_encoded=encode_and_new(pd.concat([df.iloc[:1],user_df],ignore_index=True)).iloc[1:]

# Predicting
if st.button("Predict Churn"):
    pred=model.predict(user_df_encoded)[0]
    prob=model.predict_proba(user_df_encoded)[0][1]
    st.markdown(f"### Prediction: {'Churn' if pred else 'No Churn'} ({prob:.2%})")

    explain_prediction(model,user_df_encoded,df_encoded.sample(100))
