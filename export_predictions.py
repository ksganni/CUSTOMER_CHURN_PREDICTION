import pandas as pd
import pickle
from src.data_preprocessing import load_data
from src.feature_engineering import encode_and_new

# Loading cleaned data
model_input_df=load_data() # Using data_preprocessing function to get the cleaned dataset.

# Applying the feature encoding and any new feature creation from the pipeline
encoded_df=encode_and_new(model_input_df)

# Separating features from the target column
X = encoded_df.drop("Churn", axis=1)

# Loading the best model that was saved
with open("models/best_model.pkl", "rb") as f:
    loaded = pickle.load(f)
    if isinstance(loaded, tuple):
        model = loaded[0]  
    else:
        model = loaded

# Predicting Churn labels
predicted_labels = model.predict(X)

# Getting churn probabilities. Use predict_proba if available
if hasattr(model, "predict_proba"):
    churn_probabilities = model.predict_proba(X)[:, 1]  
else:
    churn_probabilities = [None] * len(predicted_labels)

# Adding the prediction results to the original dataframe
output_df=model_input_df.copy()
output_df["PredictedChurn"]=predicted_labels
output_df["ChurnProbability"]=churn_probabilities

# Exporting to CSV
output_df.to_csv("data/predictions_output.csv",index=False)
print("Predictions exported successfully")
