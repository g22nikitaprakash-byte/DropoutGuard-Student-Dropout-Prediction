import joblib
import shap
import pandas as pd
import numpy as np
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "prediction_model.pkl"

DATA_PATH = BASE_DIR.parent / "dataset" / "data.csv"


# =========================================================
# MODEL FEATURES
# =========================================================

FEATURE_NAMES = [

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


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading model...")

model = joblib.load(MODEL_PATH)

print(
    "Model loaded:",
    type(model).__name__
)


# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep=";"
)

print(
    "Dataset shape:",
    df.shape
)


# =========================================================
# PREPARE BACKGROUND DATA
# =========================================================

X = df[FEATURE_NAMES].copy()

# Use a small background sample for SHAP
background = X.sample(
    n=min(100, len(X)),
    random_state=42
)


# =========================================================
# SAMPLE STUDENT
# =========================================================

sample_student = pd.DataFrame(

    [[

        20,

        120,

        5,

        10,

        5,

        10,

        0,

        0,

        1

    ]],

    columns=FEATURE_NAMES
)


# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(
    sample_student
)[0]


probabilities = model.predict_proba(
    sample_student
)[0]


classes = model.classes_


print("\n========================================")

print("PREDICTION RESULT")

print("========================================")

print(
    "Prediction:",
    prediction
)


print("\nClass Probabilities:")

for class_name, probability in zip(
    classes,
    probabilities
):

    print(
        f"{class_name}: "
        f"{probability * 100:.2f}%"
    )


# =========================================================
# SHAP TREE EXPLAINER
# =========================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(
    model
)


shap_values = explainer.shap_values(
    sample_student
)


print("SHAP values calculated.")


# =========================================================
# GET DROPOUT CLASS INDEX
# =========================================================

dropout_index = list(classes).index(
    "Dropout"
)


# =========================================================
# HANDLE SHAP OUTPUT
# =========================================================

if isinstance(shap_values, list):

    # Older SHAP versions:
    # list[class] -> samples x features

    feature_values = shap_values[
        dropout_index
    ][0]

else:

    # Newer SHAP versions

    values = np.array(shap_values)

    if values.ndim == 3:

        # samples x features x classes

        feature_values = values[
            0,
            :,
            dropout_index
        ]

    else:

        feature_values = values[0]


# =========================================================
# FEATURE IMPACT TABLE
# =========================================================

importance = pd.DataFrame({

    "Feature":
        FEATURE_NAMES,

    "SHAP_Value":
        feature_values,

    "Absolute_Impact":
        np.abs(feature_values)

})


importance = importance.sort_values(

    by="Absolute_Impact",

    ascending=False

)


# =========================================================
# DISPLAY EXPLANATION
# =========================================================

print("\n========================================")

print(
    "WHY DID THE MODEL PREDICT DROPOUT?"
)

print("========================================\n")


for _, row in importance.iterrows():

    shap_value = row["SHAP_Value"]


    if shap_value > 0:

        impact = "Increases dropout risk"

    elif shap_value < 0:

        impact = "Decreases dropout risk"

    else:

        impact = "No significant impact"


    print(

        f"{row['Feature']}: "
        f"{shap_value:.4f} "
        f"({impact})"

    )


# =========================================================
# TOP 5 FEATURES
# =========================================================

print("\n========================================")

print("TOP 5 IMPORTANT FEATURES")

print("========================================\n")


top_features = importance.head(5)


for index, (_, row) in enumerate(
    top_features.iterrows(),
    start=1
):

    print(

        f"{index}. "
        f"{row['Feature']} "
        f"-> "
        f"{row['SHAP_Value']:.4f}"

    )


print("\n========================================")

print(
    "SHAP explanation generated successfully!"
)

print("========================================")