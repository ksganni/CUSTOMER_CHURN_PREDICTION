# Evaluating and training the model

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_models(X_train, y_train):
    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver='lbfgs')
        ),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        results[name] = {
            "roc_auc_mean": scores.mean(),
            "roc_auc_std": scores.std()
        }

    return results


# Hyperparameter tuning and best model training
def tune_and_train_best(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Saving both model and column order
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump((best_model, X_train.columns.tolist()), f)

    return best_model, grid.best_params_
