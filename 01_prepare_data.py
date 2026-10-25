import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD ENGINEERED DATASET
# ============================================================

print("=" * 60)
print("LOADING ENGINEERED DATASET")
print("=" * 60)

df = pd.read_csv("dataset/engineered_dataset.csv")

print(f"Dataset shape: {df.shape}")
print()


# ============================================================
# 2. DISPLAY TARGET DISTRIBUTION
# ============================================================

print("=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df["Target"].value_counts())
print()


# ============================================================
# 3. REMOVE DATA LEAKAGE FEATURES
# ============================================================

# These features are derived risk/outcome indicators.
# They should NOT be directly given to the ML model.

excluded_features = [
    "Target",
    "Risk_Category",
    "Overall_Student_Risk_Score",
    "Academic_Risk_Score"
]

X = df.drop(columns=excluded_features)
y = df["Target"]


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()

print("=" * 60)
print("FEATURE INFORMATION")
print("=" * 60)

print(f"Total input features: {X.shape[1]}")
print(f"Categorical features: {len(categorical_columns)}")
print(f"Numerical features: {len(numerical_columns)}")
print()

print("Categorical Columns:")
print(categorical_columns)
print()

print("Numerical Columns:")
print(numerical_columns)
print()


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. DISPLAY TRAIN-TEST INFORMATION
# ============================================================

print("=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print()

print("Training Target Distribution:")
print(y_train.value_counts())
print()

print("Testing Target Distribution:")
print(y_test.value_counts())
print()


# ============================================================
# 7. SAVE TRAINING AND TESTING DATA
# ============================================================

X_train.to_csv(
    "dataset/X_train.csv",
    index=False
)

X_test.to_csv(
    "dataset/X_test.csv",
    index=False
)

y_train.to_csv(
    "dataset/y_train.csv",
    index=False
)

y_test.to_csv(
    "dataset/y_test.csv",
    index=False
)


# ============================================================
# 8. FINAL MESSAGE
# ============================================================

print("=" * 60)
print("ML DATA PREPARATION COMPLETED")
print("=" * 60)

print()
print("Files created successfully:")
print("1. dataset/X_train.csv")
print("2. dataset/X_test.csv")
print("3. dataset/y_train.csv")
print("4. dataset/y_test.csv")
print()
print("You are ready for the ML model training stage.")