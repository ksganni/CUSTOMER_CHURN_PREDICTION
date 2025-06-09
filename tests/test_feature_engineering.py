from src.feature_engineering import encode_and_new
import pandas as pd

def test_encoding_basic():
    df = pd.DataFrame({
        'gender': ['Male', 'Female'],
        'SeniorCitizen': ['Yes', 'No'],
        'Partner': ['Yes', 'No'],
        'Dependents': ['No', 'Yes'],
        'tenure': [1, 20],
        'MonthlyCharges': [30, 70],
        'TotalCharges': [30, 1400],
        'Churn': ['Yes', 'No']
    })

    df_encoded = encode_and_new(df)

    # Basic checks
    assert "Churn" in df_encoded.columns
    assert df_encoded["Churn"].isin([0, 1]).all()

    # Checking new feature ChargesPerMonth exists and is correct
    expected = df["TotalCharges"] / (df["tenure"] + 1)
    expected.name = "ChargesPerMonth"  # Fix: set name
    assert "ChargesPerMonth" in df_encoded.columns
    pd.testing.assert_series_equal(df_encoded["ChargesPerMonth"], expected)


def test_encoding_with_reference_columns():
    df = pd.DataFrame({
        'gender': ['Male'],
        'SeniorCitizen': ['Yes'],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [0],
        'MonthlyCharges': [50],
        'TotalCharges': [0],
        'Churn': ['No']
    })

    # Providing reference columns that include some columns not in df after encoding
    reference_columns = [
        'ChargesPerMonth',
        'gender_Male',     # created by get_dummies
        'SeniorCitizen_Yes',
        'Partner_Yes',     # missing from df, should be added as 0
        'Dependents_Yes',  # missing from df, should be added as 0
        'Churn'
    ]

    df_encoded = encode_and_new(df, reference_columns=reference_columns)

    # All reference columns should be present in the result
    assert set(reference_columns) == set(df_encoded.columns)

    # Missing columns (Partner_Yes, Dependents_Yes) should be filled with zeros
    assert (df_encoded['Partner_Yes'] == 0).all()
    assert (df_encoded['Dependents_Yes'] == 0).all()

    # ChargesPerMonth calculation
    expected_charges = df["TotalCharges"] / (df["tenure"] + 1)
    expected_charges.name = "ChargesPerMonth"  # Fix: set name
    pd.testing.assert_series_equal(df_encoded["ChargesPerMonth"], expected_charges)
