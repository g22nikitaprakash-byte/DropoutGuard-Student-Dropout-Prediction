import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. LOAD ENGINEERED DATASET
# ==========================================

data_path = "../dataset/engineered_dataset.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("DATASET LOADED")
print("=" * 60)

print("Dataset Shape:", df.shape)


# ==========================================
# 2. REMOVE DATA LEAKAGE FEATURES
# ==========================================

excluded_features = [
    "Target",
    "Risk_Category",
    "Overall_Student_Risk_Score",
    "Academic_Risk_Score"
]

X = df.drop(columns=excluded_features, errors="ignore")
y = df["Target"]


# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. IDENTIFY DATA TYPES
# ==========================================

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical Features:", len(categorical_features))
print("Numerical Features:", len(numerical_features))


# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ==========================================
# 6. DEFINE ML MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200,
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        class_weight="balanced",
        random_state=42
    )
}


# ==========================================
# 7. TRAIN MODELS
# ==========================================

results = []

best_model = None
best_model_name = None
best_f1 = -1


for name, model in models.items():

    print("\n" + "=" * 60)
    print("TRAINING:", name)
    print("=" * 60)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name


# ==========================================
# 8. SAVE RESULTS
# ==========================================

results_df = pd.DataFrame(results)

os.makedirs("../reports", exist_ok=True)

results_df.to_csv(
    "../reports/model_comparison.csv",
    index=False
)


# ==========================================
# 9. SAVE BEST MODEL
# ==========================================

os.makedirs("../ml", exist_ok=True)

joblib.dump(
    best_model,
    "../ml/best_model.pkl"
)


# ==========================================
# 10. FINAL OUTPUT
# ==========================================

print("\n")
print("=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("Best Model:", best_model_name)
print("Best Macro F1:", round(best_f1, 4))

print("\nModel comparison saved at:")
print("../reports/model_comparison.csv")

print("\nBest model saved at:")
print("../ml/best_model.pkl")

print("=" * 60)