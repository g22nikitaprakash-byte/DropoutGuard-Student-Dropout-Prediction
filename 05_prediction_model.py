import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "dataset" / "engineered_dataset.csv"
MODEL_PATH = BASE_DIR / "prediction_model.pkl"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)


# --------------------------------------------------
# FEATURES FOR PREDICTION
# --------------------------------------------------

features = [
    "Age at enrollment",
    "Admission grade",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Scholarship holder",
    "Debtor",
    "Tuition fees up to date"
]

X = df[features]
y = df["Target"]


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# RANDOM FOREST MODEL
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n----------------------------------")
print("Prediction Model Evaluation")
print("----------------------------------")

print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

print("\n----------------------------------")
print("Model saved successfully!")
print("----------------------------------")
print("Location:", MODEL_PATH)