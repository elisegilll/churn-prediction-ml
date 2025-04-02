# Import necessary libraries
 # For data splitting and model evaluation
from sklearn.model_selection import train_test_plit, cross_val_score
 # For feature scaling
from sklearn.preprocessing import StandardScaler
 # Random Forest model
from sklearn.ensemble import RandomForestClassifier
# Logistic Regression model
from sklearn.linear_model import LogisticRegression 
# For model evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score 
# For handling class imbalance
from imblearn.over_sampling import SMOTE
 # For handling missing values
from sklearn.impute import SimpleImputer
 # For data manipulation
import pandas as pd
 
# Step 1: Load the dataset
df = pd.read_csv('churndataset.csv', delimiter=';')
 
# Step 2: Clean data by replacing commas with dots and converting to numeric where needed
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.').astype(float, errors='ignore')
 
# Step 3: Separate the features (X) and target variable (y)
X = df.drop(columns=['Churn'])
y = df['Churn']
 
# Convert 'Yes'/'No' in y to binary values (1 for 'Yes' and 0 for 'No')
y = y.map({'Yes': 1, 'No': 0})
 
# Step 4: One-hot encode categorical columns to convert them to numeric
X = pd.get_dummies(X, drop_first=True)
 
# Step 5: Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
 
# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Step 7: Apply SMOTE for class balancing on the training set
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
 
# Step 8: Standardize the features
scaler = StandardScaler()
X_train_sm = scaler.fit_transform(X_train_sm)
X_test = scaler.transform(X_test)
 
# Step 9: Initialize models with increased max_iter for Logistic Regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)  # Increased max_iter
rand_forest = RandomForestClassifier(random_state=42)
 
# Step 10: Train and evaluate models
for model, name in zip([log_reg, rand_forest], ['Logistic Regression', 'Random Forest']):
    model.fit(X_train_sm, y_train_sm)
    y_pred = model.predict(X_test)
   
    print(f"\n{name} Model Performance")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
   
    cv_scores = cross_val_score(model, X_train_sm, y_train_sm, cv=10, scoring='f1')
    print(f"{name} F1-Score (10-fold CV): {cv_scores.mean():.3f}