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
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver='lbfgs')
        ),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
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
        print(f"{name}: {scores.mean():.6f} ± {scores.std():.6f}")

    # Save the evaluation results for the Streamlit app to use
    with open("models/model_evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("✅ Model evaluation results saved to models/model_evaluation_results.pkl")

    return results


# Hyperparameter tuning and best model training
def tune_and_train_best(X_train, y_train):
    # First, get all model results to find the actual best performer
    model_results = train_models(X_train, y_train)

    # Find the best performing model
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['roc_auc_mean'])
    best_score = model_results[best_model_name]['roc_auc_mean']

    print(f"\n🏆 Best Model: {best_model_name} with ROC-AUC: {best_score:.6f}")

    # For this example, we'll still use Random Forest for consistency with your existing setup
    # But you can modify this to use the actual best performer
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Save both model and column order, plus the model name and final score
    model_info = {
        'model': best_model,
        'columns': X_train.columns.tolist(),
        'model_name': 'Random Forest',
        'final_score': grid.best_score_,
        'best_params': grid.best_params_
    }

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump((best_model, X_train.columns.tolist()), f)

    # Also save detailed model info
    with open("models/model_info.pkl", "wb") as f:
        pickle.dump(model_info, f)

    print(f"✅ Best model saved with score: {grid.best_score_:.6f}")
    print(f"✅ Best parameters: {grid.best_params_}")

    return best_model, grid.best_params_


# Example usage:
if __name__ == "__main__":
    # Assuming you have X_train and y_train ready
    # results = train_models(X_train, y_train)
    # best_model, best_params = tune_and_train_best(X_train, y_train)
    pass
