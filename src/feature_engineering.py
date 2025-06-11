# Encoding and adding new features

import pandas as pd


def encode_and_new(df, reference_columns=None):
    # Avoid modifying original data
    df = df.copy()

    # Creating new feature: Charges per month
    df["ChargesPerMonth"] = df["TotalCharges"] / (df["tenure"] + 1)  # +1 to avoid division by 0

    # Encoding target if present
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    # One-hot encoding of categorical features
    categorical_columns = df.select_dtypes(include='object').columns.drop("Churn", errors="ignore")
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Aligning columns with training features
    if reference_columns is not None:
        missing_cols = set(reference_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[reference_columns]

    return df
