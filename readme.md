# 📉 CUSTOMER CHURN PREDICTION

<div align="center">

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
- [📸 App Screenshots]
- [🎯 Problem Statement](#-problem-statement)
- [📊 Data Overview](#-data-overview)
- [🗂️ Project Structure](#️-project-structure)
- [⚙️ Technical Implementation](#️-technical-implementation)
- [🛠️ Installation & Setup](#️-installation--setup)
- [📱 Usage Examples](#-usage-examples)
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

## 📸 App Screenshots

<div align="center">

### 🏠 **Homepage & Project Overview**

### 📊 **Dataset Exploration**

### 🤖 **Model Performance Comparison**

### 🔮 **Prediction Interface**

### Examples

❗ **High Risk: New Month-to-Month Customer with High Charges

⚠️ **Medium Risk: Long-Term Customer with Full Services

✅ **Low Risk: Stable Family Customer


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