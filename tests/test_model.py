import pytest
from sklearn.exceptions import NotFittedError
from src.model import train_models, tune_and_train_best
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd

# These imports help us identify which model was selected
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# ✅ Testing to check if train_models returns correct metrics for each model
def test_train_models_output_keys():
    # Loading and preprocessing the data
    df = load_data()
    df_encoded = encode_and_new(df)

    # Splitting into features and target
    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]

    # Train-test splitting + SMOTE for class balance
    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2)
    X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)

    # Training multiple models and getting evaluation results
    results = train_models(X_train_res, y_train_res)

    # Checking that each model's result has 'roc_auc_mean' and 'roc_auc_std'
    for model_name, metrics in results.items():
        assert "roc_auc_mean" in metrics
        assert "roc_auc_std" in metrics
        # AUC score should be in a valid range
        assert 0.5 <= metrics["roc_auc_mean"] <= 1.0


# ✅ Testing to check if the best model returned is trained and its parameters are valid
def test_tune_and_train_best_returns_model_and_params():
    # Load and preprocess the data
    df = load_data()
    df_encoded = encode_and_new(df)

    # Preparing training data
    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"]
    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2)
    X_train_res, y_train_res = SMOTE().fit_resample(X_train, y_train)

    # Running model tuning and training to get the best model and parameters
    best_model, best_params = tune_and_train_best(X_train_res, y_train_res)

    # ✅ Ensuring the returned model is trained (i.e., can make predictions)
    try:
        best_model.predict(X_train_res.iloc[:5])
    except NotFittedError:
        pytest.fail("Returned model is not fitted")

    # ✅ Dynamically checking best_params depending on which model was selected
    if isinstance(best_model, RandomForestClassifier):
        # RandomForest hyperparameters
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "min_samples_split" in best_params

    elif isinstance(best_model, XGBClassifier):
        # XGBoost hyperparameters
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "learning_rate" in best_params

    elif isinstance(best_model, CatBoostClassifier):
        # CatBoost hyperparameters
        assert "depth" in best_params
        assert "learning_rate" in best_params

    else:
        # Failing test if an unknown model type is returned
        pytest.fail("Unknown model type returned")


# ✅ Testing to ensure train_models throws an error if we pass empty input
def test_train_models_with_empty_data_raises():
    # Creating empty inputs
    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype=int)

    with pytest.raises(ValueError):
        train_models(X_empty, y_empty)
