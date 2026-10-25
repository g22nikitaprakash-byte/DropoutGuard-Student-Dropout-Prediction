import pandas as pd
import numpy as np

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df = pd.read_csv("../dataset/data.csv", sep=";")

print("=" * 70)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 70)

# ---------------------------------------------------
# 1. TOTAL ACADEMIC UNITS
# ---------------------------------------------------

df["Total_Credited_Units"] = (
    df["Curricular units 1st sem (credited)"]
    + df["Curricular units 2nd sem (credited)"]
)

df["Total_Enrolled_Units"] = (
    df["Curricular units 1st sem (enrolled)"]
    + df["Curricular units 2nd sem (enrolled)"]
)

df["Total_Evaluations"] = (
    df["Curricular units 1st sem (evaluations)"]
    + df["Curricular units 2nd sem (evaluations)"]
)

df["Total_Approved_Units"] = (
    df["Curricular units 1st sem (approved)"]
    + df["Curricular units 2nd sem (approved)"]
)

df["Total_Without_Evaluation"] = (
    df["Curricular units 1st sem (without evaluations)"]
    + df["Curricular units 2nd sem (without evaluations)"]
)

# ---------------------------------------------------
# 2. SEMESTER APPROVAL RATES
# ---------------------------------------------------

df["Sem1_Approval_Rate"] = np.where(
    df["Curricular units 1st sem (enrolled)"] > 0,
    df["Curricular units 1st sem (approved)"]
    / df["Curricular units 1st sem (enrolled)"],
    0
)

df["Sem2_Approval_Rate"] = np.where(
    df["Curricular units 2nd sem (enrolled)"] > 0,
    df["Curricular units 2nd sem (approved)"]
    / df["Curricular units 2nd sem (enrolled)"],
    0
)

# ---------------------------------------------------
# 3. OVERALL APPROVAL RATE
# ---------------------------------------------------

df["Overall_Approval_Rate"] = np.where(
    df["Total_Enrolled_Units"] > 0,
    df["Total_Approved_Units"]
    / df["Total_Enrolled_Units"],
    0
)

# ---------------------------------------------------
# 4. EVALUATION RATES
# ---------------------------------------------------

df["Sem1_Evaluation_Rate"] = np.where(
    df["Curricular units 1st sem (enrolled)"] > 0,
    df["Curricular units 1st sem (evaluations)"]
    / df["Curricular units 1st sem (enrolled)"],
    0
)

df["Sem2_Evaluation_Rate"] = np.where(
    df["Curricular units 2nd sem (enrolled)"] > 0,
    df["Curricular units 2nd sem (evaluations)"]
    / df["Curricular units 2nd sem (enrolled)"],
    0
)

# ---------------------------------------------------
# 5. ACADEMIC PERFORMANCE
# ---------------------------------------------------

df["Average_Semester_Grade"] = (
    df["Curricular units 1st sem (grade)"]
    + df["Curricular units 2nd sem (grade)"]
) / 2

df["Semester_Grade_Improvement"] = (
    df["Curricular units 2nd sem (grade)"]
    - df["Curricular units 1st sem (grade)"]
)

# Absolute difference between admission and later performance
df["Grade_Gap"] = (
    df["Admission grade"]
    - df["Average_Semester_Grade"]
)

# ---------------------------------------------------
# 6. CREDIT COMPLETION RATE
# ---------------------------------------------------

df["Credit_Completion_Rate"] = np.where(
    df["Total_Enrolled_Units"] > 0,
    df["Total_Credited_Units"]
    / df["Total_Enrolled_Units"],
    0
)

# ---------------------------------------------------
# 7. PROGRESSION SCORE
# ---------------------------------------------------

df["Progression_Score"] = (
    df["Credit_Completion_Rate"] * 0.4
    + df["Overall_Approval_Rate"] * 0.4
    + (
        df["Average_Semester_Grade"]
        / 20
    ) * 0.2
)

# ---------------------------------------------------
# 8. FINANCIAL RISK INDICATOR
# ---------------------------------------------------

df["Financial_Risk_Indicator"] = (
    df["Debtor"] +
    (1 - df["Tuition fees up to date"])
)

# ---------------------------------------------------
# 9. SCHOLARSHIP SUPPORT
# ---------------------------------------------------

df["Scholarship_Financial_Support"] = (
    df["Scholarship holder"]
)

# ---------------------------------------------------
# 10. ACADEMIC RISK SCORE
# ---------------------------------------------------

df["Academic_Risk_Score"] = (
    (1 - df["Overall_Approval_Rate"]) * 0.35
    + (1 - df["Credit_Completion_Rate"]) * 0.25
    + (
        df["Total_Without_Evaluation"]
        / df["Total_Enrolled_Units"].replace(0, 1)
    ) * 0.20
    + (
        (20 - df["Average_Semester_Grade"])
        / 20
    ) * 0.20
)

# ---------------------------------------------------
# 11. OVERALL STUDENT RISK SCORE
# ---------------------------------------------------

financial_component = (
    df["Financial_Risk_Indicator"] / 2
)

df["Overall_Student_Risk_Score"] = (
    df["Academic_Risk_Score"] * 0.70
    + financial_component * 0.30
)

# ---------------------------------------------------
# 12. RISK CATEGORY
# ---------------------------------------------------

df["Risk_Category"] = pd.cut(
    df["Overall_Student_Risk_Score"],
    bins=[-np.inf, 0.30, 0.60, np.inf],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# ---------------------------------------------------
# SAVE ENGINEERED DATASET
# ---------------------------------------------------

output_path = "../dataset/engineered_dataset.csv"

df.to_csv(output_path, index=False)

# ---------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------

print("\nOriginal Number of Columns: 37")
print("New Number of Columns:", len(df.columns))

print("\nNew Features Created:")

new_features = [
    "Total_Credited_Units",
    "Total_Enrolled_Units",
    "Total_Evaluations",
    "Total_Approved_Units",
    "Total_Without_Evaluation",
    "Sem1_Approval_Rate",
    "Sem2_Approval_Rate",
    "Overall_Approval_Rate",
    "Sem1_Evaluation_Rate",
    "Sem2_Evaluation_Rate",
    "Average_Semester_Grade",
    "Semester_Grade_Improvement",
    "Grade_Gap",
    "Credit_Completion_Rate",
    "Progression_Score",
    "Financial_Risk_Indicator",
    "Scholarship_Financial_Support",
    "Academic_Risk_Score",
    "Overall_Student_Risk_Score",
    "Risk_Category"
]

for feature in new_features:
    print("-", feature)

print("\nRisk Category Distribution:")
print(df["Risk_Category"].value_counts())

print("\nDataset saved to:")
print(output_path)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)