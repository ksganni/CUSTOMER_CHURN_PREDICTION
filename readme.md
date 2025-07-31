<div align="center">

# 📉 CUSTOMER CHURN PREDICTION

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com/)

**A comprehensive machine learning project that predicts whether telecom customers are likely to cancel their service (churn). This solution helps businesses identify at-risk customers and take proactive measures to retain them, ultimately reducing customer loss and boosting profitability.**

🚀 **[Try the Live Demo](https://customer-churn-prediction-8hui.onrender.com/)**

</div>

---

## 📋 Table of Contents

- [✨ Key Features](#-key-features)
- [🎯 Problem Statement](#-problem-statement)
- [📊 Data Overview](#-data-overview)
- [🗂️ Project Structure](#️-project-structure)
- [⚙️ Technical Implementation](#️-technical-implementation)
- [🛠️ Installation & Setup](#️-installation--setup)
- [📱 Usage Examples](#-usage-examples)
- [📸 App Screenshots](#-app-screenshots)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Key Features

<div align="center">
<table>
<tr>
<td align="center">🧹</td>
<td><strong>Data Cleaning & Feature Engineering</strong><br/>Advanced preprocessing and feature creation</td>
</tr>
<tr>
<td align="center">🤖</td>
<td><strong>Machine learning model training and evaluation</strong><br/>Multiple algorithms with performance comparison</td>
</tr>
<tr>
<td align="center">🪄</td>
<td><strong>Explainable AI</strong><br/>SHAP-powered prediction explanations</td>
</tr>
<tr>
<td align="center">🌐</td>
<td><strong>Interactive Web App</strong><br/>User-friendly Streamlit interface</td>
</tr>
<tr>
<td align="center">✅</td>
<td><strong>Automated Testing</strong><br/>CI/CD with GitHub Actions</td>
</tr>
<tr>
<td align="center">🐳</td>
<td><strong>Docker Ready</strong><br/>Containerized for easy deployment</td>
</tr>
</table>
</div>

---

## ❓ What is Customer Churn?

Customer churn refers to customers stopping use of a company's services or products. In telecom, this means customers canceling phone, internet, or cable services and switching to competitors.

---

## 🎯 Problem Statement

Telecom companies face significant revenue loss due to churn. This project addresses the challenge by:

✅ **Analyzing customer behavior patterns**  
✅ **Building predictive models to forecast churn**  
✅ **Providing explanations for why customers might leave**  
✅ **Creating an interactive prediction tool for business users**  
✅ **Offering actionable insights to help retention teams focus their efforts**

**Our goal:** Predict churn and explain each prediction using machine learning.

---

## 📊 Data Overview

### 📥 **Dataset Information**

| **Attribute** | **Value** |
|---------------|-----------|
| **📄 Dataset** | Telco Customer Churn Dataset from Kaggle |
| **📊 Format** | CSV file |
| **🎯 Target** | Churn column (Yes/No) |
| **📝 Link** | [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) |

### 🔍 **Feature Categories**

<details>
<summary><strong>🧑‍💼 Customer Information</strong></summary>

- **Gender:** Male or Female
- **SeniorCitizen:** Whether the customer is 65 or older (No/Yes)
- **Partner:** Whether the customer has a partner (Yes/No)
- **Dependents:** Whether the customer has dependents (Yes/No)

</details>

<details>
<summary><strong>🛠️ Services Used</strong></summary>

- **PhoneService:** Whether the customer has phone service (Yes/No)
- **MultipleLines:** Whether the customer has multiple phone lines (Yes/No/No phone service)
- **InternetService:** Type of internet service (DSL/Fiber optic/No)
- **OnlineSecurity:** Whether the customer has online security add-on (Yes/No/No internet service)
- **OnlineBackup:** Whether the customer has online backup service (Yes/No/No internet service)
- **DeviceProtection:** Whether the customer has device protection (Yes/No/No internet service)
- **TechSupport:** Whether the customer has tech support (Yes/No/No internet service)
- **StreamingTV:** Whether the customer has streaming TV (Yes/No/No internet service)
- **StreamingMovies:** Whether the customer has streaming movies (Yes/No/No internet service)

</details>

<details>
<summary><strong>💵 Billing Information</strong></summary>

- **PaperlessBilling:** Whether the customer uses paperless billing (Yes/No)
- **PaymentMethod:** How the customer pays (Electronic check/Mailed check/Bank transfer/Credit card)
- **MonthlyCharges:** Amount charged to the customer monthly (numeric)
- **TotalCharges:** Total amount charged to the customer (numeric)
- **Tenure:** Number of months the customer has been with the company (numeric)

</details>

<details>
<summary><strong>📝 Contract Information</strong></summary>

- **Contract:** Type of customer contract (Month-to-month/One year/Two year)
- **Tenure:** Number of months the customer has been with the company (numeric)

</details>

---

## 🗂️ Project Structure

```
Customer-Churn-Prediction/
│
├── .github/workflows/          # 🔄 Automated testing and deployment
│   ├── ci.yml                 # Runs tests when code is pushed to GitHub
│   └── docker.yml             # Builds Docker images for deployment
│
├── App/                       # 🌐 Web application files
│   ├── assets/                # Images and static files for the app
│   ├── page_modules/          # Individual pages of the web app
│   │   ├── about.py          # Homepage with project information
│   │   ├── dataset.py        # Page to explore the dataset
│   │   ├── models.py         # Page showing model performance comparison
│   │   └── predictor.py      # Interactive prediction page
│   ├── shap_helper.py        # Code for explaining model predictions
│   └── streamlit_app.py      # Main application entry point
│
├── data/                     # 📊 All data files
│   ├── original_dataset.csv  # Raw customer data
│   └── predictions_output.csv # Results from model predictions
│
├── Models/                   # 🤖 Trained machine learning models
│   ├── best_model.pkl        # The final selected model
│   └── model_evaluation_results.pkl # Performance metrics
│
├── notebook/                 # 📓 Jupyter notebook for analysis
│   └── customer_churn_prediction.ipynb # Complete data analysis workflow
│
├── src/                      # 🔧 Core Python modules
│   ├── data_preprocessing.py # Data cleaning and preparation
│   ├── feature_engineering.py # Creating new features from existing data
│   └── model.py             # Model training and evaluation
│
└── tests/                   # ✅ Automated tests to ensure code quality
    ├── test_data_preprocessing.py
    ├── test_feature_engineering.py
    └── test_model.py
```

---

## ⚙️ Technical Implementation

### 1️⃣ **Data Preprocessing** (`src/data_preprocessing.py`)

**Purpose:** Clean and prepare raw data for machine learning

**What it does:**
- **Handles missing values:** Some customers have missing TotalCharges data, which we estimate using MonthlyCharges × Tenure
- **Fixes data types:** Converts TotalCharges from text to numbers for mathematical operations
- **Standardizes formats:** Converts SeniorCitizen from numbers (0/1) to text (No/Yes) for consistency
- **Removes unnecessary columns:** Eliminates customerID since it doesn't help predict churn
- **Validates data quality:** Checks for inconsistencies and outliers

**Example transformation:**
```python
# Before: TotalCharges = " " (empty string)
# After:  TotalCharges = 45.2 * 12 = 542.4 (calculated from MonthlyCharges and tenure)
```

### 2️⃣ **Feature Engineering** (`src/feature_engineering.py`)

**Purpose:** Create new, more informative features from existing data

**What it does:**
- **Creates spending efficiency metric:** ChargesPerMonth = TotalCharges / (tenure + 1) to identify high-value customers
- **Encodes categorical variables:** Converts text categories to numbers that machine learning models can understand
    - "Yes" → 1, "No" → 0
    - "Male" → 1, "Female" → 0
- **Handles multiple categories:** Uses one-hot encoding for features with more than two options
- **Ensures consistency:** Makes sure test data has the same format as training data

**Example of one-hot encoding:**
```python
# Original: Contract = "Month-to-month"
# Becomes: Contract_Month-to-month = 1, Contract_One_year = 0, Contract_Two_year = 0
```

### 3️⃣ **Exploratory Data Analysis (EDA)**

**Purpose:** Understand patterns in the data before building models

**Key analyses performed:**

**📊 Class Distribution Analysis**
- **Churn rate:** Approximately 26.5% of customers churn
- **Imbalance handling:** Used SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset
- **Impact:** Prevents model from being biased toward predicting "No Churn" all the time

**🔗 Feature Correlation Analysis**
- **Heatmaps:** Visualize relationships between different features
- **Key findings:**
    - Short tenure strongly correlates with churn
    - Month-to-month contracts have higher churn rates
    - Fiber optic internet users churn more than DSL users
    - Higher monthly charges correlate with increased churn

**📈 Distribution Analysis**
- **Tenure distribution:** New customers (0-12 months) are most likely to churn
- **Contract type impact:** Month-to-month contracts show 42% churn vs. 11% for longer contracts
- **Service usage patterns:** Customers without add-on services churn more frequently

### 4️⃣ **Model Training & Selection** (`src/model.py`)

**Purpose:** Build and compare multiple machine learning models to find the best predictor

**🤖 Models Tested:**

| Model | Description |
|-------|-------------|
| **Logistic Regression** | - Simple, interpretable model<br>- Good baseline for binary classification<br>- Fast training and prediction |
| **Decision Tree** | - Easy to understand decision rules<br>- Can capture non-linear relationships<br>- Prone to overfitting |
| **Random Forest** | - Combines multiple decision trees<br>- Reduces overfitting through ensemble learning<br>- Provides feature importance rankings |
| **XGBoost** | - Advanced boosting algorithm<br>- Excellent performance on structured data<br>- Handles missing values automatically |
| **CatBoost** | - Specialized for categorical features<br>- Reduces need for extensive preprocessing<br>- Built-in overfitting protection |


**📊 Model Evaluation Process**

- **Cross-Validation:** 5-fold cross-validation was used to ensure robust and unbiased performance estimates.
- **Metrics Used:**
  - ROC-AUC (Receiver Operating Characteristic – Area Under Curve)
  - Accuracy Score

**🎛️ Hyperparameter Tuning**

- **Technique:** `GridSearchCV` systematically tested combinations of hyperparameters to optimize model performance.
- **Best Model Tuned:** **Random Forest**
- **Hyperparameters Tuned:**
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [10, 20, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

**Model Selection Results**
The following models were trained and evaluated using 5-fold cross-validation:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- CatBoost

**🏆 Final Selected Model: Random Forest**

| Metric       | Score             |
|--------------|-------------------|
| **ROC-AUC**  | **0.928 ± 0.051** |
| **Accuracy** | **0.848 ± 0.067** |

- **Why this model?**  
  Random Forest achieved the highest performance on both ROC-AUC and Accuracy metrics, indicating strong predictive power and generalizability.

- **Training Data:** 31 engineered and encoded features were used for training.


**🧠 Metric Explanations**

**🎯 ROC-AUC (Receiver Operating Characteristic – Area Under Curve)**
- Measures the model’s ability to distinguish between churners and non-churners.
- Particularly useful for imbalanced datasets like churn.
- Scores interpretation:
  - 0.5 → No discrimination (random guess)
  - 1.0 → Perfect discrimination
  - 0.8 → Excellent performance

**✔️ Accuracy**
- Measures the proportion of correct predictions.
- Can be misleading on imbalanced datasets.
- Best used alongside ROC-AUC to evaluate overall model effectiveness.


### 5️⃣ **Model Explanation with SHAP** (`App/shap_helper.py`)

**Purpose:** Explain exactly why the model made its prediction

**🔍 What SHAP provides:**
- **Feature impact:** Shows how each customer characteristic influenced the prediction
- **Direction of influence:** Whether each feature pushed toward "Churn" or "Stay"
- **Magnitude of impact:** How strongly each feature influenced the decision

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

---

## 🛠️ Installation & Setup

### 🐳 **Option 1: Using Docker (Recommended)**

```bash
# Clone the repository
git clone https://github.com/ksganni/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Build and run with Docker
docker build -t churn-prediction .
docker run -p 8501:8501 churn-prediction
```

### 🐍 **Option 2: Local Installation**

```bash
# Clone the repository
git clone https://github.com/ksganni/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run App/streamlit_app.py
```

### 🚀 **Option 3: Deploy to Render**

1. **Fork this repository** to your GitHub account
2. **Connect to Render:**
   - Go to [Render.com](https://render.com) and sign up
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
3. **Configure deployment:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run App/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
   - **Environment:** Python 3.8+
4. **Deploy** and your app will be live!

### 📋 **Requirements**

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
catboost>=1.2.0
shap>=0.42.0
plotly>=5.15.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

---

## 📱 Usage Examples

### 💡 **Sample Scenarios To Check Churn Risk**

#### ❗ **High Risk: Short-Term Customer Overpaying**
```
Gender: Female
Senior Citizen: Yes
Partner: No
Dependents: No
Tenure: 5 months
Phone Service: Yes
Multiple Lines: Yes
Internet Service: Fiber Optic
Online Security: No
Online Backup: No
Device Protection: No
Tech Support: No
Streaming TV: No
Streaming Movies: No
Contract: Month-to-Month
Paperless Billing: No
Payment Method: Mailed Check
Monthly Charges: $95.00
Total Charges: $475.00
```

#### ⚠️ **Medium Risk: Moderate Tenure with Some Stability**
```
Gender: Female
Senior Citizen: No
Partner: Yes
Dependents: No
Tenure: 15 months
Phone Service: Yes
Multiple Lines: Yes
Internet Service: Fiber Optic
Online Security: No
Online Backup: Yes
Device Protection: No
Tech Support: No
Streaming TV: Yes
Streaming Movies: No
Contract: Month-to-Month
Paperless Billing: Yes
Payment Method: Credit Card(Automatic)
Monthly Charges: $78.90
Total Charges: $1,183.50
```

#### ✅ **Low Risk: Premium Family Customer**
```
Gender: Male
Senior Citizen: No
Partner: Yes
Dependents: Yes
Tenure: 65 months
Phone Service: Yes
Multiple Lines: Yes
Internet Service: Fiber Optic
Online Security: Yes
Online Backup: Yes
Device Protection: Yes
Tech Support: Yes
Streaming TV: Yes
Streaming Movies: Yes
Contract: Two Year
Paperless Billing: Yes
Payment Method: Bank Transfer(Automatic)
Monthly Charges: $95.20
Total Charges: $6188.00
```

---

## 📸 App Screenshots

### 🏠 **Homepage & Project Overview**

<div align="center">
<img width="1438" height="688" alt="Screenshot 2025-07-09 at 3 52 14 PM" src="https://github.com/user-attachments/assets/ea46ca97-9381-4163-82ae-5092434c1a6f" /><img width="1438" height="474" alt="Screenshot 2025-07-09 at 3 52 44 PM" src="https://github.com/user-attachments/assets/350b5aa4-828a-4d1d-88d2-9deb85148ca0" /><img width="1438" height="592" alt="Screenshot 2025-07-09 at 3 53 05 PM" src="https://github.com/user-attachments/assets/5e0671df-a1ab-47e9-90f8-1998d208cc3a" /><img width="1438" height="249" alt="Screenshot 2025-07-09 at 3 53 21 PM" src="https://github.com/user-attachments/assets/e9934703-1553-4350-bfa5-50544ff8178e" /><img width="1438" height="119" alt="Screenshot 2025-07-09 at 3 53 56 PM" src="https://github.com/user-attachments/assets/1e0798a7-b292-4502-9da1-b4376e6befc4" /><img width="1439" height="625" alt="Screenshot 2025-07-09 at 3 54 27 PM" src="https://github.com/user-attachments/assets/abf96c59-a841-456e-8151-57b53ae185f6" /><img width="1439" height="625" alt="Screenshot 2025-07-09 at 3 54 38 PM" src="https://github.com/user-attachments/assets/340f63b8-29c1-4d59-9c37-4152d6050612" /><img width="1439" height="625" alt="Screenshot 2025-07-09 at 3 54 49 PM" src="https://github.com/user-attachments/assets/1925191e-6ead-4692-a351-8ae8c1e6ff86" />

</div>

### 📊 **Dataset Exploration**

<div align="center">

<img width="1439" height="687" alt="Screenshot 2025-07-09 at 3 55 13 PM" src="https://github.com/user-attachments/assets/4d983560-8967-49aa-8e66-26cf352cea1d" /><img width="1439" height="545" alt="Screenshot 2025-07-09 at 3 55 42 PM" src="https://github.com/user-attachments/assets/b2e2a399-f664-45cb-a52c-f4e5084be2c2" /><img width="1439" height="589" alt="Screenshot 2025-07-09 at 3 56 06 PM" src="https://github.com/user-attachments/assets/7867fa8f-00b5-4daf-816e-7764f2fa7a96" />

</div>

### 🤖 **Model Performance Comparison**

<div align="center">

<img width="1439" height="529" alt="Screenshot 2025-07-09 at 3 57 57 PM" src="https://github.com/user-attachments/assets/dd9b0fd3-18a3-456c-8aa9-7feded36e9ce" /><img width="1439" height="537" alt="Screenshot 2025-07-09 at 3 58 12 PM" src="https://github.com/user-attachments/assets/54985b5e-53a9-4b8c-b196-70ffd30188f6" /><img width="1439" height="521" alt="Screenshot 2025-07-09 at 3 58 25 PM" src="https://github.com/user-attachments/assets/1006d192-2966-4bc2-b35c-4baa2523c064" /><img width="1439" height="635" alt="Screenshot 2025-07-09 at 3 58 51 PM" src="https://github.com/user-attachments/assets/4961af31-f572-4b3b-bb8c-cfea2662f0c0" />

</div>

### 🔮 **Prediction Interface**

<div align="center">

<img width="1438" height="328" alt="Screenshot 2025-07-09 at 3 22 52 PM" src="https://github.com/user-attachments/assets/59a19b93-55b0-4a0f-9bb1-c8f89ef24708" /><img width="1438" height="605" alt="Screenshot 2025-07-09 at 3 23 21 PM" src="https://github.com/user-attachments/assets/cef7e61c-f050-48bf-bd17-721912cc87e5" /><img width="1438" height="600" alt="Screenshot 2025-07-09 at 3 23 38 PM" src="https://github.com/user-attachments/assets/1cc065f9-10a1-4c0d-a5e8-5e2767b11f94" /><img width="1438" height="600" alt="Screenshot 2025-07-09 at 3 23 50 PM" src="https://github.com/user-attachments/assets/a6595248-7d65-402a-8b7d-a1dd772bd125" />

</div>

### Examples

❗ **High Risk: New Month-to-Month Customer with High Charges**

<div align="center">

<img width="1439" height="555" alt="Screenshot 2025-07-09 at 7 52 51 PM" src="https://github.com/user-attachments/assets/9e7c0e09-1a27-451b-87e5-4454e1179539" /><img width="1439" height="593" alt="Screenshot 2025-07-09 at 7 53 21 PM" src="https://github.com/user-attachments/assets/bf195716-dcbc-4464-8033-f646fcc9c8d8" /><img width="1439" height="576" alt="Screenshot 2025-07-09 at 7 53 36 PM" src="https://github.com/user-attachments/assets/631ba123-1c63-4d71-ba04-9b22e5d24837" /><img width="1440" height="412" alt="Screenshot 2025-07-31 at 3 02 21 PM" src="https://github.com/user-attachments/assets/a1a2fd39-40c8-418a-9b9e-253ec5df4c31" /><img width="1440" height="592" alt="Screenshot 2025-07-31 at 3 02 53 PM" src="https://github.com/user-attachments/assets/8113d981-b2b4-4e07-8162-b4deaa5daf92" /><img width="1440" height="463" alt="Screenshot 2025-07-31 at 3 03 08 PM" src="https://github.com/user-attachments/assets/bec19ee8-36cb-42b4-ae9a-4d5e703d6e2c" /><img width="1440" height="451" alt="Screenshot 2025-07-31 at 3 03 24 PM" src="https://github.com/user-attachments/assets/904abef5-a4aa-468a-a47c-f95c475a98cc" /><img width="1440" height="189" alt="Screenshot 2025-07-31 at 3 03 39 PM" src="https://github.com/user-attachments/assets/b36c75a3-e197-4f84-81f1-5d4f30305596" /><img width="1440" height="257" alt="Screenshot 2025-07-31 at 3 03 53 PM" src="https://github.com/user-attachments/assets/81f1c8dd-8546-4821-aa91-f6c426737fd8" />

</div>

⚠️ **Medium Risk: Long-Term Customer with Full Services**

<div align="center">

<img width="1439" height="563" alt="Screenshot 2025-07-09 at 7 59 48 PM" src="https://github.com/user-attachments/assets/ea1a5334-755f-4128-8284-d918b68b7aff" /><img width="1439" height="587" alt="Screenshot 2025-07-09 at 8 00 19 PM" src="https://github.com/user-attachments/assets/93a7242e-ac4b-451e-a523-03838057935e" /><img width="1439" height="562" alt="Screenshot 2025-07-09 at 8 00 36 PM" src="https://github.com/user-attachments/assets/0c7806d8-7868-4da2-bcae-bca19eb9fab5" /><img width="1440" height="443" alt="Screenshot 2025-07-31 at 3 07 13 PM" src="https://github.com/user-attachments/assets/dffde107-92b5-4cde-9019-ac0b9dee0088" /><img width="1440" height="592" alt="Screenshot 2025-07-31 at 3 07 28 PM" src="https://github.com/user-attachments/assets/c7a1867c-18a7-4532-bc8e-d7be1c63382d" /><img width="1440" height="456" alt="Screenshot 2025-07-31 at 3 07 45 PM" src="https://github.com/user-attachments/assets/5e7d2335-15a1-43c9-b886-869eb11b1437" /><img width="1440" height="447" alt="Screenshot 2025-07-31 at 3 07 57 PM" src="https://github.com/user-attachments/assets/da25ef8a-33f5-4b62-9e69-c46f5485e878" /><img width="1440" height="203" alt="Screenshot 2025-07-31 at 3 08 13 PM" src="https://github.com/user-attachments/assets/9a49355f-52f4-4257-a364-009853c45af6" /><img width="1440" height="214" alt="Screenshot 2025-07-31 at 3 08 40 PM" src="https://github.com/user-attachments/assets/3a1c1745-58ac-4b0a-9d94-2ee34f99c653" />

</div>

✅ **Low Risk: Established Senior Customer**

<div align="center">

<img width="1440" height="555" alt="Screenshot 2025-07-31 at 3 28 02 PM" src="https://github.com/user-attachments/assets/35b74886-a539-4935-afff-a36129ba2076" /><img width="1440" height="587" alt="Screenshot 2025-07-31 at 3 28 17 PM" src="https://github.com/user-attachments/assets/7a35d362-492f-46e1-828b-a60e6b10b57a" /><img width="1440" height="562" alt="Screenshot 2025-07-31 at 3 28 54 PM" src="https://github.com/user-attachments/assets/bded6ea4-8366-4290-a369-2a92df5458cc" /><img width="1440" height="440" alt="Screenshot 2025-07-31 at 3 29 06 PM" src="https://github.com/user-attachments/assets/b1d78f74-f823-43fe-a93a-2c74ef548b67" /><img width="1440" height="595" alt="Screenshot 2025-07-31 at 3 29 21 PM" src="https://github.com/user-attachments/assets/d7660a0a-953a-464b-a614-e12bb8542e54" /><img width="1440" height="475" alt="Screenshot 2025-07-31 at 3 29 33 PM" src="https://github.com/user-attachments/assets/ddaef122-de68-4eb4-9a08-144aa376cb91" /><img width="1440" height="441" alt="Screenshot 2025-07-31 at 3 29 47 PM" src="https://github.com/user-attachments/assets/2153cf4f-2786-42b8-8282-7a37a9f02704" /><img width="1440" height="196" alt="Screenshot 2025-07-31 at 3 29 57 PM" src="https://github.com/user-attachments/assets/d5f4ffb5-62cd-4091-9dce-0367bbcb32a8" /><img width="1440" height="207" alt="Screenshot 2025-07-31 at 3 30 05 PM" src="https://github.com/user-attachments/assets/82578195-4c46-459c-9f00-4d106c8daecd" />

</div>

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### 📝 **How to Contribute:**

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data) from Kaggle
- **SHAP:** For model explainability
- **Streamlit:** For the web application framework
- **Scikit-learn:** For machine learning algorithms
- **Render:** For reliable cloud deployment

---

<div align="center">



**Developed with 🛠️ and 📊**

</div>
