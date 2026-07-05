import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ==========================================
# 1. SIMULATE / LOAD TELCO CHURN DATASET
# ==========================================
# Replace this section with pd.read_csv('your_telco_churn_data.csv') in practice
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'tenure': np.random.randint(1, 72, n_samples),
    'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples),
    'TotalCharges': np.random.choice([np.nan, ' '], p=[0.01, 0.99], size=n_samples), # Simulating raw string/missing values
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer'], n_samples),
    'Churn': np.random.choice(['No', 'Yes'], p=[0.73, 0.27], size=n_samples)
})

# Realistically, 'TotalCharges' often contains blank spaces that need converting to numeric
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].replace(' ', np.nan), errors='coerce')
# Fill missing simulated TotalCharges with a placeholder for illustration
data['TotalCharges'] = data['TotalCharges'].fillna(data['tenure'] * data['MonthlyCharges'])

# Separate Features (X) and Target (y)
X = data.drop(columns=['Churn'])
y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0) # Encode target as binary

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ==========================================
# 2. DEFINE PREPROCESSING PIPELINES
# ==========================================
# Identify column types dynamically
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['Contract', 'InternetService', 'PaymentMethod']

# Numeric Pipeline: Impute missing values then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Pipeline: Impute missing values then One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # drop='first' avoids multicollinearity
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)


# ==========================================
# 3. BUILD THE FULL MODEL PIPELINE
# ==========================================
# Create a generic pipeline with a placeholder classifier step
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42)) # Placeholder
])


# ==========================================
# 4. HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ==========================================
# We define a search grid that tests BOTH Logistic Regression and Random Forest
param_grid = [
    {
        'classifier': [LogisticRegression(max_iter=1000, random_state=42)],
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['liblinear', 'lbfgs']
    },
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    }
]

# Instantiate GridSearchCV targeting ROC-AUC optimize for Churn imbalance
grid_search = GridSearchCV(
    estimator=full_pipeline, 
    param_grid=param_grid, 
    cv=5, 
    scoring='roc_auc', 
    n_jobs=-1, 
    verbose=1
)

print("Starting Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)

print(f"\nBest Model Parameters Found: {grid_search.best_params_}")
print(f"Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")


# ==========================================
# 5. EVALUATE THE BEST PIPELINE
# ==========================================
best_pipeline = grid_search.best_estimator_

# Predict on test data
y_pred = best_pipeline.predict(X_test)
y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

print("\n=== Model Performance Evaluation ===")
print(classification_report(y_test, y_pred))
print(f"Test ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")


# ==========================================
# 6. EXPORT THE PRODUCTION-READY PIPELINE
# ==========================================
# This saves everything (Imputers, Scalers, Encoders, and Model weights) into one file
model_filename = 'telco_churn_production_pipeline.joblib'
joblib.dump(best_pipeline, model_filename)
print(f"\nSuccessfully exported the end-to-end ML pipeline to: '{model_filename}'")

# --- Verification of Reusability ---
# Simulating a production environment reading raw data
loaded_pipeline = joblib.load(model_filename)
sample_raw_data = X_test.iloc[[0]] # Mimic an incoming raw single API payload

# Predict directly without manual preprocessing
sample_prediction = loaded_pipeline.predict(sample_raw_data)
sample_proba = loaded_pipeline.predict_proba(sample_raw_data)[:, 1]

print("\n--- Production Verification Pass ---")
print(f"Input Raw Data:\n{sample_raw_data.to_string()}\n")
print(f"Predicted Class: {sample_prediction[0]} (Probability of Churn: {sample_proba[0]:.2%})")