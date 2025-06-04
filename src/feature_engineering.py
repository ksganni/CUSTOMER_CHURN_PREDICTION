# Encoding and adding new features

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_and_new(df):

    # Creating new feature: Charges per month
    df["ChargesPerMonth"]=df["TotalCharges"]/(df["tenure"]+1) # +1 to avoid div by 0

    # Labelling encode target variable
    df["Churn"]=df["Churn"].map({"No":0,"Yes":1})

    # One-hot encoding of categorical features
    categorical_columns=df.select_dtypes(include='object').columns.drop("Churn")
    df=pd.get_dummies(df,columns=categorical_columns,drop_first=True)

    return df

