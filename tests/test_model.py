import os
import pickle
import pytest
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new
from src.model import tune_and_train_best

def test_tune_and_train_best(tmp_path):
    df = load_data()
    df_encoded = encode_and_new(df)

    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]

    best_model, best_params = tune_and_train_best(X, y)

    assert hasattr(best_model, "predict")
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert "min_samples_split" in best_params

    model_file = tmp_path / "best_model.pkl"

    with open("models/best_model.pkl", "rb") as f:
        saved_model, columns = pickle.load(f)

    assert hasattr(saved_model, "predict")
    assert isinstance(columns, list)
    assert all(isinstance(col, str) for col in columns)
