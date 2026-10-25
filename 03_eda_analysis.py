import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../dataset/data.csv", sep=";")

print("=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ---------------------------------------------------
# 1. Target Distribution
# ---------------------------------------------------

target_counts = df["Target"].value_counts()

plt.figure(figsize=(8, 5))
target_counts.plot(kind="bar")
plt.title("Student Outcome Distribution")
plt.xlabel("Student Outcome")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../reports/01_target_distribution.png")
plt.show()

# ---------------------------------------------------
# 2. Age Distribution
# ---------------------------------------------------

plt.figure(figsize=(8, 5))
df["Age at enrollment"].plot(kind="hist", bins=20)
plt.title("Age Distribution at Enrollment")
plt.xlabel("Age")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("../reports/02_age_distribution.png")
plt.show()

# ---------------------------------------------------
# 3. Admission Grade by Target
# ---------------------------------------------------

plt.figure(figsize=(8, 5))
df.boxplot(column="Admission grade", by="Target")
plt.title("Admission Grade by Student Outcome")
plt.suptitle("")
plt.xlabel("Student Outcome")
plt.ylabel("Admission Grade")
plt.tight_layout()
plt.savefig("../reports/03_admission_grade_by_target.png")
plt.show()

# ---------------------------------------------------
# 4. First Semester Grade by Target
# ---------------------------------------------------

plt.figure(figsize=(8, 5))
df.boxplot(
    column="Curricular units 1st sem (grade)",
    by="Target"
)
plt.title("First Semester Grade by Student Outcome")
plt.suptitle("")
plt.xlabel("Student Outcome")
plt.ylabel("First Semester Grade")
plt.tight_layout()
plt.savefig("../reports/04_first_sem_grade_by_target.png")
plt.show()

# ---------------------------------------------------
# 5. Second Semester Grade by Target
# ---------------------------------------------------

plt.figure(figsize=(8, 5))
df.boxplot(
    column="Curricular units 2nd sem (grade)",
    by="Target"
)
plt.title("Second Semester Grade by Student Outcome")
plt.suptitle("")
plt.xlabel("Student Outcome")
plt.ylabel("Second Semester Grade")
plt.tight_layout()
plt.savefig("../reports/05_second_sem_grade_by_target.png")
plt.show()

# ---------------------------------------------------
# 6. Scholarship vs Target
# ---------------------------------------------------

scholarship_table = pd.crosstab(
    df["Scholarship holder"],
    df["Target"]
)

scholarship_table.plot(kind="bar", figsize=(8, 5))

plt.title("Scholarship Status vs Student Outcome")
plt.xlabel("Scholarship Holder (0 = No, 1 = Yes)")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../reports/06_scholarship_vs_target.png")
plt.show()

# ---------------------------------------------------
# 7. Debtor vs Target
# ---------------------------------------------------

debtor_table = pd.crosstab(
    df["Debtor"],
    df["Target"]
)

debtor_table.plot(kind="bar", figsize=(8, 5))

plt.title("Debtor Status vs Student Outcome")
plt.xlabel("Debtor (0 = No, 1 = Yes)")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../reports/07_debtor_vs_target.png")
plt.show()

# ---------------------------------------------------
# 8. Tuition Fees vs Target
# ---------------------------------------------------

tuition_table = pd.crosstab(
    df["Tuition fees up to date"],
    df["Target"]
)

tuition_table.plot(kind="bar", figsize=(8, 5))

plt.title("Tuition Fee Status vs Student Outcome")
plt.xlabel("Tuition Fees Up To Date (0 = No, 1 = Yes)")
plt.ylabel("Number of Students")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../reports/08_tuition_vs_target.png")
plt.show()

# ---------------------------------------------------
# 9. Correlation Analysis
# ---------------------------------------------------

numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))
plt.imshow(correlation, aspect="auto")
plt.colorbar()

plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("../reports/09_correlation_matrix.png")
plt.show()

# ---------------------------------------------------
# 10. Print Important Statistics
# ---------------------------------------------------

print("\n--- Academic Statistics ---")

print("\nAdmission Grade:")
print(df.groupby("Target")["Admission grade"].mean())

print("\nFirst Semester Grade:")
print(
    df.groupby("Target")[
        "Curricular units 1st sem (grade)"
    ].mean()
)

print("\nSecond Semester Grade:")
print(
    df.groupby("Target")[
        "Curricular units 2nd sem (grade)"
    ].mean()
)

print("\nScholarship Status:")
print(pd.crosstab(df["Scholarship holder"], df["Target"]))

print("\nDebtor Status:")
print(pd.crosstab(df["Debtor"], df["Target"]))

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nCharts saved inside reports folder.")