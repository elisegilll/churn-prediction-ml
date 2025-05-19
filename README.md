# 📈 Customer Churn Prediction – Telecom Dataset

## Overview  
This project uses supervised machine learning techniques to predict customer churn in the telecom industry. The objective is to identify which customers are likely to leave the service provider and understand the key drivers of churn using exploratory data analysis and predictive modeling.

---

## Tools & Technologies  
- **Language**: Python  
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
- **Modeling**: Random Forest, Logistic Regression  
- **Preprocessing**: SMOTE for class balancing  
- **Evaluation**: Confusion Matrix, F1-score, ROC Curve, Cross-validation  

---

## Goals  
- Improve customer retention by identifying at-risk users  
- Understand features that strongly contribute to churn  
- Deploy reusable and interpretable models for business analysts

---

## Dataset  
Source: [Kaggle Telecom Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)  
Contains 7,043 records including customer demographics, account details, and service usage.

---

## Key Results  
- Achieved **F1-score improvement of 20%** after balancing classes with SMOTE  
- Top features influencing churn: Contract Type, Tenure, Internet Service  
- Random Forest model performed best with 10-fold cross-validation

---

## Insights  
- Month-to-month customers on fibre internet are most likely to churn  
- Customers with longer tenure and bundled services are more loyal  
- Contract length has a direct impact on retention

---

## Folder Structure  
project/
├── data/
├── notebooks/
├── models/
├── churn_model.py
├── requirements.txt
└── README.md


---

## How to Run  
```bash
git clone https://github.com/elisegilll/customer-churn-prediction
cd customer-churn-prediction
pip install -r requirements.txt
python churn_model.py



