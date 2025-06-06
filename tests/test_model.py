import pandas as pd
from src.feature_engineering import encode_and_new
from src.data_preprocessing import load_data
from src.model import train_models

def test_model_training():
    df=load_data()
    df_encoded=encode_and_new(df)

    X=df_encoded.drop("Churn",axis=1)
    y=df_encoded["Churn"]

    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    X_train,_,y_train,_=train_test_split(X,y,stratify=y,test_size=0.2)
    X_train_res,y_train_res=SMOTE().fit_resample(X_train,y_train)

    results=train_models(X_train_res,y_train_res)
    assert all("roc_auc_mean" in v for v in results.values())
