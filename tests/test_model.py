import pytest
from sklearn.exceptions import NotFittedError
from src.model import train_models, tune_and_train_best
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd


def test_train_models_output_keys():
    df = load_data()
    df_encoded = encode_and_new(df)

    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]

    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2)
    X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)

    results = train_models(X_train_res, y_train_res)

    # Checking all models return roc_auc_mean and roc_auc_std keys
    for model_name, metrics in results.items():
        assert "roc_auc_mean" in metrics
        assert "roc_auc_std" in metrics
        # roc_auc_mean should be between 0.5 and 1 (reasonable range)
        assert 0.5 <= metrics["roc_auc_mean"] <= 1.0


def test_tune_and_train_best_returns_model_and_params():
    df = load_data()
    df_encoded = encode_and_new(df)

    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]

    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2)
    X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)

    best_model, best_params = tune_and_train_best(X_train_res, y_train_res)

    # Checking returned object is a fitted model
    try:
        best_model.predict(X_train_res.iloc[:5])
    except NotFittedError:
        pytest.fail("Returned model is not fitted")

    # Checking best_params keys
    assert "n_estimators" in best_params
    assert "max_depth" in best_params
    assert "min_samples_split" in best_params


def test_train_models_with_empty_data_raises():
    # Empty DataFrame inputs
    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype=int)

    with pytest.raises(ValueError):
        train_models(X_empty, y_empty)
