from src.feature_engineering import encode_and_new
import pandas as pd

def test_encoding():
    df=pd.DataFrame({
        'gender':['Male','Female'],
        'SeniorCitizen':['Yes','No'],
        'Partner':['Yes','No'],
        'Dependents':['No','Yes'],
        'tenure':[1,20],
        'MonthlyCharges':[30,70],
        'TotalCharges':[30,1400],
        'Churn':['Yes','No']
    })

    df_encoded=encode_and_new(df)
    assert "Churn" in df_encoded.columns
    assert df_encoded["Churn"].isin([0,1]).all()