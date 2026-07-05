import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# --------------------------------------------------------------------------
# 1. Mock Dataset Creation (Simulating Telco Churn Data)
# --------------------------------------------------------------------------
# Creating a dummy dataset to make this script immediately executable.
np.random.seed(42)
n_samples = 1000

data = {
    'tenure': np.random.randint(1, 72, size=n_samples),
    'MonthlyCharges': np.random.uniform(18.0, 120.0, size=n_samples),
    'TotalCharges': np.random.uniform(18.0, 8000.0, size=n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples),
    'Churn': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Introduce a few missing values to test our pipeline's robustness
df.loc[df.sample(frac=0.05).index, 'TotalCharges'] = np.nan

# Separate features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --------------------------------------------------------------------------
# 2. Pipeline Feature Engineering & Preprocessing
# --------------------------------------------------------------------------
# Identify numerical and categorical features dynamically
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['Contract', 'PaymentMethod', 'InternetService']

# Numerical Preprocessing: Impute missing values with median, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Preprocessing: Impute missing values with 'missing', then One-Hot Encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Combine preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# --------------------------------------------------------------------------
# 3. Model Orchestration Pipeline
# --------------------------------------------------------------------------
# Build the master pipeline with a placeholder estimator
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression()) # Placeholder, overridden by GridSearch
])

# --------------------------------------------------------------------------
# 4. Hyperparameter Tuning with GridSearchCV
# --------------------------------------------------------------------------
# Define parameter grid exploring both Logistic Regression and Random Forest
param_grid = [
    {
        'classifier': [LogisticRegression(max_iter=1000, random_state=42)],
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['liblinear', 'lbfgs']
    },
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }
]

# Initialize GridSearch
grid_search = GridSearchCV(
    estimator=full_pipeline, 
    param_grid=param_grid, 
    cv=5, 
    scoring='f1', # Optimizing for F1-score due to typical churn imbalance
    n_jobs=-1,
    verbose=1
)

# Execute Search
print("Training models and tuning hyperparameters...")
grid_search.fit(X_train, y_train)

# Output results
print("\n--- Best Model Identification ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

# --------------------------------------------------------------------------
# 5. Model Validation & Exportation
# --------------------------------------------------------------------------
# Evaluate the optimal pipeline on the holdout test set
best_pipeline = grid_search.best_estimator_
test_score = best_pipeline.score(X_test, y_test)
print(f"Holdout Test Accuracy: {test_score:.4f}")

# Export the entire end-to-end pipeline (Preprocessing + Model) for production
model_filename = 'telco_churn_production_pipeline.joblib'
joblib.dump(best_pipeline, model_filename)
print(f"\nSuccess: Production pipeline exported successfully to '{model_filename}'")