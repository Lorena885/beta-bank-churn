# 🏦 Bank Customer Churn Prediction — Beta Bank

A supervised machine learning project to predict customer churn for a retail bank, enabling proactive retention campaigns before customers leave.

---

## 📌 Business Problem

Beta Bank was losing customers every month. Retaining an existing customer is significantly cheaper than acquiring a new one, so the business team needed a reliable way to **identify at-risk customers before they churn**.

This model flags customers by risk level — HIGH, MEDIUM, or LOW — so the retention team can prioritise outreach and allocate campaign resources efficiently.

---

## 🎯 Objective

Build a classification model that predicts whether a customer will leave the bank, using historical behavioural and demographic data.

**Success criteria:**
- F1 Score ≥ 0.59 on the test set
- AUC-ROC measured and compared against F1

---

## 📊 Dataset

- **Source:** [Churn Modelling — Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Records:** 10,000 customers
- **Features:** Credit score, geography, gender, age, tenure, balance, number of products, credit card status, active membership, estimated salary
- **Target:** `Exited` — 1 if the customer left the bank, 0 if they stayed
- **Class distribution:** ~80% stayed / ~20% churned (imbalanced)

---

## 🔧 Tools & Libraries

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| Data manipulation | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook / VS Code |

---

## 🧪 Methodology

### 1. Data Preparation
- Dropped non-predictive columns (`RowNumber`, `CustomerId`, `Surname`)
- Applied One-Hot Encoding to categorical variables (`Geography`, `Gender`)
- Handled 909 missing values in `Tenure` using median imputation (fitted on training set only to prevent data leakage)
- Scaled numerical features with `StandardScaler`
- Split data: **60% train / 20% validation / 20% test** (stratified)

### 2. Handling Class Imbalance
Three strategies were tested and compared:

| Strategy | F1 Score |
|---|---|
| Oversampling (Decision Tree) | ~0.52 |
| Undersampling (Decision Tree) | ~0.48 |
| Logistic Regression (balanced weights) | ~0.53 |
| **Random Forest (balanced weights)** | **~0.65** ✅ |

### 3. Hyperparameter Tuning
Grid search over `n_estimators` [100, 200, 300] and `max_depth` [8, 10, 12, 14]:
- Optimal depth: **10** — deeper trees overfit, shallower trees underfit
- Increasing tree count beyond 100 did not improve F1
- **Best config: n_estimators=100, max_depth=10, class_weight='balanced'**

---

## 📈 Results

| Metric | Validation | Test |
|---|---|---|
| F1 Score | 0.6490 | 0.5866 |
| AUC-ROC | 0.8711 | 0.8542 |

- **AUC-ROC of 0.854** confirms the model strongly distinguishes between churned and retained customers
- The model identifies **~58% of actual churners** (recall), enabling targeted intervention
- Class weight adjustment proved more effective than resampling techniques

---

## 🎯 Real-World Application

The final section of the notebook simulates production use:
- Scores every customer in the test set with a churn probability
- Classifies each customer as **HIGH RISK (≥70%)**, **MEDIUM RISK (45–70%)**, or **LOW RISK**
- Outputs a prioritised list of customers for the retention team

```
RECOMMENDED ACTIONS:
  HIGH RISK   → Immediate personal outreach
  MEDIUM RISK → Targeted email/offer campaign
  LOW RISK    → Standard engagement programme
```

---

## 📁 Project Structure

```
beta-bank-churn/
│
├── beta_bank_churn.ipynb   # Main notebook — full analysis and model
├── Churn.csv               # Dataset (from Kaggle)
└── README.md               # Project documentation
```

---

## 🚀 How to Run

1. Clone this repository:
```bash
git clone https://github.com/Lorena885/beta-bank-churn.git
cd beta-bank-churn
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) and place `Churn.csv` in the project folder.

4. Open and run the notebook:
```bash
jupyter notebook beta_bank_churn.ipynb
```

---

## 👩‍💻 Author

**Lorena Cardona**  
Data Science | Operations & Management  
📍 Auckland, New Zealand  
🔗 [LinkedIn](https://linkedin.com/in/lorenacarvajalc) | [Portfolio](https://datascienceportfol.io/lcarvajalcar)
