import zipfile

zip_path = "/Users/krishnasathvikaganni/PROJECTS/CUSTOMER_CHURN_PREDICTION/data/telco-customer-churn-prediction-dataset.zip"
output_path = "/Users/krishnasathvikaganni/PROJECTS/CUSTOMER_CHURN_PREDICTION/data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_path)

print("Successfully unzipped the dataset.")
