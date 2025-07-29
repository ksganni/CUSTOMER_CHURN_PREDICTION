import pytest
from sklearn.exceptions import NotFittedError
from src.model import train_models, tune_and_train_best
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def test_train_models_output_keys():
    df = load_data()
    df_encoded = encode_and_new(df)

    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]

    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2)
    X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)

    results = train_models(X_train_res, y_train_res)

    for model_name, metrics in results.items():
        assert "roc_auc_mean" in metrics
        assert "roc_auc_std" in metrics
        assert 0.5 <= metrics["roc_auc_mean"] <= 1.0


def test_tune_and_train_best_returns_model_and_params():
    df = load_data()
    df_encoded = encode_and_new(df)

    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]

    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2)
    X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)

    best_model, best_params = tune_and_train_best(X_train_res, y_train_res)

    try:
        best_model.predict(X_train_res.iloc[:5])
    except NotFittedError:
        pytest.fail("Returned model is not fitted")

    model_type = type(best_model)

    if model_type == RandomForestClassifier:
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "min_samples_split" in best_params

    elif model_type == XGBClassifier:
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "learning_rate" in best_params

    elif model_type == CatBoostClassifier:
        assert "depth" in best_params
        assert "learning_rate" in best_params

    else:
        pytest.fail(f"Unsupported model type: {model_type}")


def test_train_models_with_empty_data_raises():
    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype=int)

    with pytest.raises(ValueError):
        train_models(X_empty, y_empty)
