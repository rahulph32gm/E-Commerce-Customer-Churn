
# E-Commerce Customer Churn Risk Predictor

An end-to-end Machine Learning web application designed to identify high-risk e-commerce customers and prevent churn. This project transitions from rigorous baseline analysis to an advanced Stacking Ensemble framework, deployed with an interactive interface.

## Why I Chose This Dataset?

Customer retention is very important for any e-commerce business. Finding a new customer is 5 times more costly than keeping an old one. I chose this dataset with 6,000 customer rows because it has good data about customer behavior, money spent, and satisfaction scores. It is perfect to practice my machine learning skills on a real business problem.

 **Important Note:** The original dataset did not have any target variable (like 'Churn'). So, I used rule-based feature engineering with Pandas to create our own 'Churn' column based on customer complaints and last purchase days. This makes the project more challenging and realistic!

---

## 📊 Project Performance Summary

| Model Configuration                   | Accuracy         | F1-Score         | Recall (Class 1) | Precision (Class 1) |
| :------------------------------------ | :--------------- | :--------------- | :--------------- | :------------------ |
| **GridSearchCV (Baseline LR)**  | 82.92%           | 0.6908           | 56.97%           | 87.74%              |
| **Stacking Classifier (Final)** | **83.50%** | **0.7122** | **60.95%** | **85.66%**    |

### Core Strategy & Key Improvements:

* **Balanced Equilibrium:** The Stacking Ensemble successfully closed the precision-recall variance gap, lifting the F1-score from `0.69` to `0.71`.
* **Robust Recall Lift:** True positive churn detection expanded significantly, raising the recall from **57% to 61%**, successfully containing false negatives amidst noise.

---

## The Architecture Blueprint

### 1. Preprocessing Pipeline (`scikit-learn`)

* **Categorical Handling:** `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore")`.
* **Numerical Scaling:** `SimpleImputer(strategy="median")` paired with robust `StandardScaler()`.

### 2. Stacking Ensemble Framework

* **Base Learners:**
  * Logistic Regression (`C=0.1`, `L1 penalty`, `liblinear`)
  * AdaBoost Classifier (`n_estimators=200`, `lr=1.0`)
  * Gradient Boosting Classifier (`n_estimators=100`, `max_depth=3`, `lr=0.2`)
* **Meta-Learner:** Optimized `RandomForestClassifier` (`n_estimators=200`, `max_depth=10`, `sqrt`).

---

## Tech Stack Used

* **Languages:** Python
* **ML Libraries:** Scikit-Learn, Pandas, NumPy, XGBoost
* **Visualizations:** Seaborn, Matplotlib
* **Deployment & UI:** Streamlit, Joblib

---

## How to Run Locally

#### Install required dependencies:

* pip install streamlit pandas scikit-learn joblib xgboost

#### Launch the web application:

* streamlit run app.py

#### **Clone the repository:**
```bash
git clone [https://github.com/rahulph32gm/E-Commerce-Customer-Churn.git](https://github.com/rahulph32gm/E-Commerce-Customer-Churn.git)
cd E-Commerce-Customer-Churn
```
