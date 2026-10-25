import pandas as pd
import joblib
import os

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD BEST MODEL
# ==========================================

model_path = "best_model.pkl"

model = joblib.load(model_path)

print("=" * 60)
print("BEST MODEL LOADED")
print("=" * 60)


# ==========================================
# 2. LOAD TEST DATA
# ==========================================

X_test = pd.read_csv("../dataset/X_test.csv")
y_test = pd.read_csv("../dataset/y_test.csv")

# y_test CSV may contain one column
y_test = y_test.iloc[:, 0]


print("Test Data Shape:", X_test.shape)


# ==========================================
# 3. MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 4. CLASSIFICATION REPORT
# ==========================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# 5. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ==========================================
# 6. SAVE CONFUSION MATRIX
# ==========================================

os.makedirs("../reports", exist_ok=True)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.title("Best Model - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "../reports/confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 7. FINAL MESSAGE
# ==========================================

print("\n")
print("=" * 60)
print("MODEL EVALUATION COMPLETED")
print("=" * 60)

print("Confusion Matrix saved:")
print("../reports/confusion_matrix.png")

print("=" * 60)