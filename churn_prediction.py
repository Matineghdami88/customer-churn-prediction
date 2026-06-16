# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("churn.csv")

print("Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# CLEAN DATA
# ==========================================

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# ==========================================
# TARGET
# ==========================================

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# ==========================================
# FEATURES / TARGET
# ==========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# ==========================================
# NUMERICAL / CATEGORICAL COLUMNS
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

# ==========================================
# PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ==========================================
# LOGISTIC REGRESSION PIPELINE
# ==========================================

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)

# ==========================================
# TRAIN LOGISTIC REGRESSION
# ==========================================

print("\nTraining Logistic Regression...")

logistic_pipeline.fit(
    X_train,
    y_train
)

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = logistic_pipeline.predict(X_test)

y_prob = logistic_pipeline.predict_proba(
    X_test
)[:, 1]

# ==========================================
# EVALUATION
# ==========================================

print("\n===== LOGISTIC REGRESSION =====")

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    "Precision:",
    precision_score(y_test, y_pred)
)

print(
    "Recall:",
    recall_score(y_test, y_pred)
)

print(
    "F1 Score:",
    f1_score(y_test, y_pred)
)

print(
    "ROC AUC:",
    roc_auc_score(y_test, y_prob)
)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# ==========================================
# RANDOM FOREST PIPELINE
# ==========================================

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=42
            )
        )
    ]
)

# ==========================================
# TRAIN RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

rf_pipeline.fit(
    X_train,
    y_train
)

# ==========================================
# RANDOM FOREST PREDICTIONS
# ==========================================

rf_pred = rf_pipeline.predict(X_test)

rf_prob = rf_pipeline.predict_proba(
    X_test
)[:, 1]

# ==========================================
# RANDOM FOREST EVALUATION
# ==========================================

print("\n===== RANDOM FOREST =====")

print(
    "Accuracy:",
    accuracy_score(y_test, rf_pred)
)

print(
    "Precision:",
    precision_score(y_test, rf_pred)
)

print(
    "Recall:",
    recall_score(y_test, rf_pred)
)

print(
    "F1 Score:",
    f1_score(y_test, rf_pred)
)

print(
    "ROC AUC:",
    roc_auc_score(y_test, rf_prob)
)

print("\nClassification Report")

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

transformed_features = (
    rf_pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

importances = (
    rf_pipeline.named_steps["classifier"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": transformed_features,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features")

print(
    importance_df.head(20)
)

# ==========================================
# SAVE MODEL
# ==========================================

import joblib

joblib.dump(
    rf_pipeline,
    "churn_model.pkl"
)

print("\nModel saved as churn_model.pkl")