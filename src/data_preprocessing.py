import pandas as pd
import os


# Loading the dataset
def load_data():
    file_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(file_path)

    # Dropping the customerID as it is not useful for prediction
    if 'customerID' in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Converting TotalCharges to numeric, coerce errors
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Filling the missing TotalCharges values with MonthlyCharges * tenure
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])

    # Converting SeniorCitizen from 0/1 to 'No'/'Yes'
    if df['SeniorCitizen'].dtype in [int, float]:
        df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})

    # Stripping extra whitespaces from column names
    df.columns = df.columns.str.strip()

    return df
