# 📉 CUSTOMER CHURN PREDICTION

A comprehensive machine learning project that predicts whether telecom customers are likely to cancel their service (churn). This solution helps businesses identify at-risk customers and take proactive measures to retain them, ultimately reducing customer loss and boosting profitability.

**✨ Key features include:**

🧹 Data cleaning and feature engineering

🤖 Machine learning model training and evaluation

🪄 Prediction with explainability using SHAP

🌐 A user-friendly web interface built with Streamlit

✅ Automated testing and CI with GitHub Actions

🐳 Docker containerization for deployment


## ❓ What is Customer Churn?

Customer churn refers to customers stopping use of a company's services or products. In telecom, this means customers canceling phone, internet, or cable services and switching to competitors.

## 🎯 Problem Statement

Telecom companies face significant revenue loss due to churn. This project addresses the challenge by:
✅ Analyzing customer behavior patterns
✅ Building predictive models to forecast churn
✅ Providing explanations for why customers might leave
✅ Creating an interactive prediction tool for business users
✅ Offering actionable insights to help retention teams focus their efforts

**Our goal:** Predict churn and explain each prediction using machine learning.

## 📊 Data Overview

#### Data Source:

**📥 Dataset:** Telco Customer Churn Dataset from Kaggle
**📄 Format:** CSV file
**🎯 Target:** Churn column (Yes/No)
**📝 Link:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data)

#### Feature Categories:

##### 🧑‍💼 Customer Info:

- **Gender:** Male or Female
- **SeniorCitizen:** Whether the customer is 65 or older (No/Yes)
- **Partner:** Whether the customer has a partner (Yes/No)
- **Dependents:** Whether the customer has dependents (Yes/No)

##### 🛠️ Services Used:

- **PhoneService:** Whether the customer has phone service (Yes/No)
- **MultipleLines:** Whether the customer has multiple phone lines (Yes/No/No phone service)
- **InternetService:** Type of internet service (DSL/Fiber optic/No)
- **OnlineSecurity:** Whether the customer has online security add-on (Yes/No/No internet service)
- **OnlineBackup:** Whether the customer has online backup service (Yes/No/No internet service)
- **DeviceProtection:** Whether the customer has device protection (Yes/No/No internet service)
- **TechSupport:** Whether the customer has tech support (Yes/No/No internet service)
- **StreamingTV:** Whether the customer has streaming TV (Yes/No/No internet service)
- **StreamingMovies:** Whether the customer has streaming movies (Yes/No/No internet service)

##### 💵 Billing Info:

- **PaperlessBilling:** Whether the customer uses paperless billing (Yes/No)
- **PaymentMethod:** How the customer pays (Electronic check/Mailed check/Bank transfer/Credit card)
- **MonthlyCharges:** Amount charged to the customer monthly (numeric)
- **TotalCharges:** Total amount charged to the customer (numeric)
- **Tenure:** Number of months the customer has been with the company (numeric)

##### 📝 Contract Info

- **Contract:** Type of customer contract (Month-to-month/One year/Two year)
- **Tenure:** Number of months the customer has been with the company (numeric)

## 🗂️ Project Structure

```
Customer-Churn-Prediction/
│
├── .github/workflows/          # Automated testing and deployment
│   ├── ci.yml                 # Runs tests when code is pushed to GitHub
│   └── docker.yml             # Builds Docker images for deployment
│
├── App/                       # Web application files
│   ├── assets/                # Images and static files for the app
│   ├── page_modules/          # Individual pages of the web app
│   │   ├── about.py          # Homepage with project information
│   │   ├── dataset.py        # Page to explore the dataset
│   │   ├── models.py         # Page showing model performance comparison
│   │   └── predictor.py      # Interactive prediction page
│   ├── shap_helper.py        # Code for explaining model predictions
│   └── streamlit_app.py      # Main application entry point
│
├── data/                     # All data files
│   ├── original_dataset.csv  # Raw customer data
│   └── predictions_output.csv # Results from model predictions
│
├── Models/                   # Trained machine learning models
│   ├── best_model.pkl        # The final selected model
│   └── model_evaluation_results.pkl # Performance metrics
│
├── notebook/                 # Jupyter notebook for analysis
│   └── customer_churn_prediction.ipynb # Complete data analysis workflow
│
├── src/                      # Core Python modules
│   ├── data_preprocessing.py # Data cleaning and preparation
│   ├── feature_engineering.py # Creating new features from existing data
│   └── model.py             # Model training and evaluation
│
└── tests/                   # Automated tests to ensure code quality
    ├── test_data_preprocessing.py
    ├── test_feature_engineering.py
    └── test_model.py
```

## ⚙️ Technical Implementation

##### 1️⃣ Data Preprocessing (src/data_preprocessing.py)

**Purpose:** Clean and prepare raw data for machine learning

**What it does:**

- **Handles missing values:** Some customers have missing TotalCharges data, which we estimate using MonthlyCharges × Tenure
- **Fixes data types:** Converts TotalCharges from text to numbers for mathematical operations
- **Standardizes formats:** Converts SeniorCitizen from numbers (0/1) to text (No/Yes) for consistency
- **Removes unnecessary columns:** Eliminates customerID since it doesn't help predict churn
- **Validates data quality:** Checks for inconsistencies and outliers

**Example transformation:**

```
Before: TotalCharges = " " (empty string)
After:  TotalCharges = 45.2 * 12 = 542.4 (calculated from MonthlyCharges and tenure)
```

##### 2️⃣ Feature Engineering (src/feature_engineering.py)

**Purpose:** Create new, more informative features from existing data

**What it does:**

- **Creates spending efficiency metric:** ChargesPerMonth = TotalCharges / (tenure + 1) to identify high-value customers
- **Encodes categorical variables:** Converts text categories to numbers that machine learning models can understand
    - "Yes" → 1, "No" → 0
    - "Male" → 1, "Female" → 0
- **Handles multiple categories:** Uses one-hot encoding for features with more than two options
    - PaymentMethod_Electronic_check → 1 if Electronic check, 0 otherwise
    - PaymentMethod_Mailed_check → 1 if Mailed check, 0 otherwise
- **Ensures consistency:** Makes sure test data has the same format as training data

**Example of one-hot encoding:**

```
Original: Contract = "Month-to-month"
Becomes: Contract_Month-to-month = 1, Contract_One_year = 0, Contract_Two_year = 0
```

##### 3️⃣ Exploratory Data Analysis (EDA)

**Purpose:** Understand patterns in the data before building models

**Key analyses performed:**

**Class Distribution Analysis**

- **Churn rate:** Approximately 26.5% of customers churn
- **Imbalance handling:** Used SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset
- **Impact:** Prevents model from being biased toward predicting "No Churn" all the time

**Feature Correlation Analysis**

- **Heatmaps:** Visualize relationships between different features
- **Key findings:**
    - Short tenure strongly correlates with churn
    - Month-to-month contracts have higher churn rates
    - Fiber optic internet users churn more than DSL users
    - Higher monthly charges correlate with increased churn

**Distribution Analysis**

- **Tenure distribution:** New customers (0-12 months) are most likely to churn
- **Contract type impact:** Month-to-month contracts show 42% churn vs. 11% for longer contracts
- **Service usage patterns:** Customers without add-on services churn more frequently

##### 4️⃣ Model Training & Selection (src/model.py)

**Purpose:** Build and compare multiple machine learning models to find the best predictor

**Models Tested**

**1. Logistic Regression**
    - Simple, interpretable model
    - Good baseline for binary classification
    - Fast training and prediction
**2. Decision Tree**
    - Easy to understand decision rules
    - Can capture non-linear relationships
    - Prone to overfitting
**3. Random Forest**
    - Combines multiple decision trees
    - Reduces overfitting through ensemble learning
    - Provides feature importance rankings
**4. XGBoost (Extreme Gradient Boosting)**
    - Advanced boosting algorithm
    - Excellent performance on structured data
    - Handles missing values automatically
**5. CatBoost (Categorical Boosting)**
    - Specialized for categorical features
    - Reduces need for extensive preprocessing
    - Built-in overfitting protection

**Model Evaluation Process**

- **Cross-validation:** 5-fold cross-validation ensures robust performance estimates
- **Metric used:** ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
ROC-AUC = 0.5: Random guessing (useless model)
ROC-AUC = 1.0: Perfect predictions (rarely achievable)
ROC-AUC > 0.8: Generally considered good performance

**Hyperparameter Tuning**

- **GridSearchCV:** Systematically tests different parameter combinations
- **Parameters tuned for Random Forest:**
    - **n_estimators:** Number of trees (50, 100, 200)
    - **max_depth:** Maximum tree depth (10, 20, None)
    - **min_samples_split:** Minimum samples to split a node (2, 5, 10)
    - **min_samples_leaf:** Minimum samples in leaf nodes (1, 2, 4)

**Model Selection Results**

Random Forest was selected as the best model with:

- **ROC-AUC Score:** 0.847 (excellent performance)
- **Consistency:** Stable performance across different data splits
- **Interpretability:** Provides clear feature importance rankings
- **Robustness:** Less prone to overfitting than single decision trees

##### 5️⃣ Model Evaluation & Visualization (App/page_modules/models.py)

**Purpose:** Present model performance comparisons in an understandable format

**Features:**

- **Performance table:** Shows ROC-AUC scores for all tested models
- **Ranking visualization:** Bar charts and horizontal rankings
- **Best model highlighting:** Clearly identifies the selected model
- **Performance explanation:** Describes what the scores mean in business terms

##### 6️⃣ Prediction & Explanation (App/page_modules/predictor.py)

**Purpose:** Explain exactly why the model made its prediction

**What SHAP provides:**

**Feature impact:** Shows how each customer characteristic influenced the prediction
**Direction of influence:** Whether each feature pushed toward "Churn" or "Stay"
**Magnitude of impact:** How strongly each feature influenced the decision

**SHAP visualizations:**

**📊 Bar Chart:** Shows top features that influenced the prediction
**📋 Detailed Table:** Lists all features with their SHAP values
**📝 Summary:** Explains the prediction in plain English

**Example SHAP explanation:**

```
Customer is LIKELY TO CHURN (78% probability)

Top factors increasing churn risk:
+ Month-to-month contract: +0.23 (very strong indicator)
+ Tenure only 3 months: +0.18 (new customers often leave)
+ No online security: +0.12 (missing valuable services)

Top factors decreasing churn risk:
- Has partner: -0.08 (stable family situation)
- Low monthly charges: -0.05 (price-conscious customer)
```

**Fallback Mechanism**
If SHAP fails (due to unusual input combinations), the system provides:
- **Feature importance:** Shows which features generally matter most for predictions
- **General explanation:** Describes typical patterns without customer-specific details

