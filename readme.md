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

**📊 Model Evaluation Process:**
- **Cross-validation:** 5-fold cross-validation ensures robust performance estimates
- **Metric used:** ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
  - ROC-AUC = 0.5: Random guessing (useless model)
  - ROC-AUC = 1.0: Perfect predictions (rarely achievable)
  - ROC-AUC > 0.8: Generally considered good performance

**🎛️ Hyperparameter Tuning**

- **GridSearchCV:** Systematically tests different parameter combinations
- **Parameters tuned for Random Forest:**
    - **n_estimators:** Number of trees (50, 100, 200)
    - **max_depth:** Maximum tree depth (10, 20, None)
    - **min_samples_split:** Minimum samples to split a node (2, 5, 10)
    - **min_samples_leaf:** Minimum samples in leaf nodes (1, 2, 4)

**🏆 Model Selection Results:**

**Selected Model:** CatBoost
- **ROC-AUC Score:** 0.932 (outstanding performance)
- **Why this model?** CatBoost achieved the highest ROC-AUC score, indicating excellent performance in distinguishing between churning and non-churning customers
- **Cross-Validation:** All models were evaluated using 5-fold cross-validation
- **Training Features:** 31 features


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

#### ✅ **Low Risk: Ultra Long-Term Loyal Customer**
```
Gender: Male
Senior Citizen: No
Partner: Yes
Dependents: Yes
Tenure: 36 months
Phone Service: Yes
Multiple Lines: Yes
Internet Service: DSL
Online Security: Yes
Online Backup: Yes
Device Protection: Yes
Tech Support: Yes
Streaming TV: No
Streaming Movies: Yes
Contract: Two Years
Paperless Billing: Yes
Payment Method: Credit Card(Automatic)
Monthly Charges: $72.40
Total Charges: $2,606.40
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

<img width="1439" height="555" alt="Screenshot 2025-07-09 at 7 52 51 PM" src="https://github.com/user-attachments/assets/9e7c0e09-1a27-451b-87e5-4454e1179539" /><img width="1439" height="593" alt="Screenshot 2025-07-09 at 7 53 21 PM" src="https://github.com/user-attachments/assets/bf195716-dcbc-4464-8033-f646fcc9c8d8" /><img width="1439" height="576" alt="Screenshot 2025-07-09 at 7 53 36 PM" src="https://github.com/user-attachments/assets/631ba123-1c63-4d71-ba04-9b22e5d24837" /><img width="1439" height="459" alt="Screenshot 2025-07-09 at 7 54 08 PM" src="https://github.com/user-attachments/assets/0b35eb26-98ba-42ef-a2e9-b24af0ce8c11" /><img width="1439" height="597" alt="Screenshot 2025-07-09 at 7 54 40 PM" src="https://github.com/user-attachments/assets/7982bec6-8469-48b1-a5f4-2d71d974ec64" /><img width="1439" height="495" alt="Screenshot 2025-07-09 at 7 55 05 PM" src="https://github.com/user-attachments/assets/5547629a-75eb-4daa-9215-d1a6b0d2f67e" /><img width="1439" height="447" alt="Screenshot 2025-07-09 at 7 55 51 PM" src="https://github.com/user-attachments/assets/11260de8-63e6-48ad-912a-91d53cbf861f" /><img width="1439" height="209" alt="Screenshot 2025-07-09 at 7 56 09 PM" src="https://github.com/user-attachments/assets/398f4580-d120-4ed7-b6c2-f81c607033e4" /><img width="1439" height="244" alt="Screenshot 2025-07-09 at 7 56 31 PM" src="https://github.com/user-attachments/assets/af404ce6-e82a-41a5-8131-b1b900faa722" />

</div>

⚠️ **Medium Risk: Long-Term Customer with Full Services**

<div align="center">

<img width="1439" height="563" alt="Screenshot 2025-07-09 at 7 59 48 PM" src="https://github.com/user-attachments/assets/ea1a5334-755f-4128-8284-d918b68b7aff" /><img width="1439" height="587" alt="Screenshot 2025-07-09 at 8 00 19 PM" src="https://github.com/user-attachments/assets/93a7242e-ac4b-451e-a523-03838057935e" /><img width="1439" height="562" alt="Screenshot 2025-07-09 at 8 00 36 PM" src="https://github.com/user-attachments/assets/0c7806d8-7868-4da2-bcae-bca19eb9fab5" /><img width="1439" height="438" alt="Screenshot 2025-07-09 at 8 00 48 PM" src="https://github.com/user-attachments/assets/1357253e-32dd-4310-9a47-5dc3fc880def" /><img width="1439" height="598" alt="Screenshot 2025-07-09 at 8 01 03 PM" src="https://github.com/user-attachments/assets/09a5618d-ce7a-4671-abd0-359dd6835c35" /><img width="1439" height="460" alt="Screenshot 2025-07-09 at 8 01 40 PM" src="https://github.com/user-attachments/assets/f0a2d6ae-c455-48f4-a41b-7bd33a8c736e" /><img width="1439" height="452" alt="Screenshot 2025-07-09 at 8 02 22 PM" src="https://github.com/user-attachments/assets/b58aa001-3483-4162-8466-27bcac763eb9" /><img width="1439" height="199" alt="Screenshot 2025-07-09 at 8 02 45 PM" src="https://github.com/user-attachments/assets/82848a3f-a204-457b-b553-4687ba1e388e" /><img width="1439" height="208" alt="Screenshot 2025-07-09 at 8 02 58 PM" src="https://github.com/user-attachments/assets/85fb9751-00de-4746-9b4b-719367c00c37" />

</div>

✅ **Low Risk: Stable Family Customer**

<div align="center">

<img width="1439" height="555" alt="Screenshot 2025-07-09 at 8 05 43 PM" src="https://github.com/user-attachments/assets/aafe2df0-0361-4ea1-b3db-b9118fbdc405" /><img width="1439" height="586" alt="Screenshot 2025-07-09 at 8 05 59 PM" src="https://github.com/user-attachments/assets/079ba1d1-f9d2-48d0-a9da-4ac3a81fda2e" /><img width="1439" height="570" alt="Screenshot 2025-07-09 at 8 06 26 PM" src="https://github.com/user-attachments/assets/b095c308-a299-4206-8309-4da194a76141" /><img width="1439" height="446" alt="Screenshot 2025-07-09 at 8 06 39 PM" src="https://github.com/user-attachments/assets/c9318afd-2030-4f61-8c26-2e3125b7d354" /><img width="1439" height="591" alt="Screenshot 2025-07-09 at 8 07 04 PM" src="https://github.com/user-attachments/assets/dc93d62c-15c1-41c5-85a6-da3a85e27f0d" /><img width="1439" height="477" alt="Screenshot 2025-07-09 at 8 07 17 PM" src="https://github.com/user-attachments/assets/c0d91683-7b1a-4ec4-9bc1-eeecc9468c29" /><img width="1439" height="448" alt="Screenshot 2025-07-09 at 8 07 30 PM" src="https://github.com/user-attachments/assets/9a155963-6045-4b0d-9f9e-3de050c994c6" /><img width="1439" height="195" alt="Screenshot 2025-07-09 at 8 07 52 PM" src="https://github.com/user-attachments/assets/8f47caea-cbd7-43e9-a8ab-b50c6797a223" /><img width="1439" height="251" alt="Screenshot 2025-07-09 at 8 08 01 PM" src="https://github.com/user-attachments/assets/3cf2b18e-e3cd-4d37-95de-cd63c57e99bb" />

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
