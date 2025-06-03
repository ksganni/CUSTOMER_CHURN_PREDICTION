import pandas as pd
import os

# Loading the dataset
def load_data():
    file_path=os.path.join("data","WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df=pd.read_csv(file_path)

    # Stripping spaces from column names
    df.columns=df.columns.str.strip()

    # Converting Totalcharges to numeric, coerce errors and drop the missing
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df=df.dropna(subset=['TotalCharges'])

    # Dropping customerID 
    if 'customerID' in df.columns:
        df=df.drop(columns=['customerID'])

    return df
