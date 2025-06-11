from src.feature_engineering import encode_and_new
from src.data_preprocessing import load_data
import pickle
import shap


def test_shap_explanation_runs():
    df = encode_and_new(load_data())
    X = df.drop("Churn", axis=1)

    with open("models/best_model.pkl", "rb") as f:
        loaded = pickle.load(f)
        if isinstance(loaded, tuple):
            model = loaded[0]
        else:
            model = loaded

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.sample(1))

    assert shap_values is not None
