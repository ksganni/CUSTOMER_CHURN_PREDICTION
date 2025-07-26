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
        roc_auc_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        accuracy_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        
        results[name] = {
            "roc_auc_mean": roc_auc_scores.mean(),
            "roc_auc_std": roc_auc_scores.std(),
            "accuracy_mean": accuracy_scores.mean(),
            "accuracy_std": accuracy_scores.std()
        }

        print(f"{name}:")
        print(f"  ROC-AUC     : {roc_auc_scores.mean():.6f} ± {roc_auc_scores.std():.6f}")
        print(f"  Accuracy    : {accuracy_scores.mean():.6f} ± {accuracy_scores.std():.6f}\n")

    # Save the evaluation results for the Streamlit app to use
    with open("models/model_evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print("Model evaluation results saved to models/model_evaluation_results.pkl")

    return results


# Hyperparameter tuning and best model training
def tune_and_train_best(X_train, y_train, metric='roc_auc'):
    # First, get all model results to find the actual best performer
    model_results = train_models(X_train, y_train)

    key_metric = f"{metric}_mean"
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x][key_metric])
    best_score = model_results[best_model_name][key_metric]

    print(f"\nBest Model based on {metric.upper()}: {best_model_name} with score: {best_score:.6f}")

    # Define model objects and param grids
    model_objects = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver='lbfgs')
        ),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    param_grids = {
        "Logistic Regression": {
            "logisticregression__C": [0.1, 1.0, 10.0]
        },
        "Decision Tree": {
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        },
        "Random Forest": {
            "n_estimators": [100, 150],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        },
        "XGBoost": {
            "n_estimators": [100, 150],
            "max_depth": [3, 5, 7]
        },
        "CatBoost": {
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1]
        }
    }

    # Get best model object and param grid
    best_model_obj = model_objects[best_model_name]
    best_param_grid = param_grids[best_model_name]

    # Perform GridSearchCV
    grid = GridSearchCV(best_model_obj, best_param_grid, cv=5, scoring=metric)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Save both model and column order, plus the model name and final score
    model_info = {
        'model': best_model,
        'columns': X_train.columns.tolist(),
        'model_name': best_model_name,
        'final_score': grid.best_score_,
        'best_params': grid.best_params_
    }

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump((best_model, X_train.columns.tolist()), f)

    # Also save detailed model info
    with open("models/model_info.pkl", "wb") as f:
        pickle.dump(model_info, f)

    print(f"Best model ({best_model_name}) saved with {metric.upper()} score: {grid.best_score_:.6f}")
    print(f"Best parameters: {grid.best_params_}")

    return best_model, grid.best_params_


# Example usage:
if __name__ == "__main__":
    # Assuming you have X_train and y_train ready
    # results = train_models(X_train, y_train)
    # best_model, best_params = tune_and_train_best(X_train, y_train, metric='roc_auc')  # or metric='accuracy'
    pass
