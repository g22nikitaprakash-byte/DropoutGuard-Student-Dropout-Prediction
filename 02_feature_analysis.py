import pandas as pd

# Load dataset
df = pd.read_csv("../dataset/data.csv", sep=";")

print("=" * 70)
print("STUDENT DROPOUT DATASET - FEATURE ANALYSIS")
print("=" * 70)

# Basic information
print("\nDataset Shape:")
print(df.shape)

print("\nTotal Features:")
print(len(df.columns) - 1)

# Feature-by-feature analysis
print("\n" + "=" * 70)
print("FEATURE DETAILS")
print("=" * 70)

for i, column in enumerate(df.columns, start=1):

    print("\n" + "-" * 70)
    print(f"Feature {i}: {column}")
    print("Data Type:", df[column].dtype)
    print("Unique Values:", df[column].nunique())
    print("Missing Values:", df[column].isnull().sum())

    # Show range for numerical columns
    if pd.api.types.is_numeric_dtype(df[column]):
        print("Minimum:", df[column].min())
        print("Maximum:", df[column].max())
        print("Mean:", round(df[column].mean(), 2))

# Target analysis
print("\n" + "=" * 70)
print("TARGET ANALYSIS")
print("=" * 70)

print("\nTarget Counts:")
print(df["Target"].value_counts())

print("\nTarget Percentages:")
print((df["Target"].value_counts(normalize=True) * 100).round(2))

# Save feature summary
summary = pd.DataFrame({
    "Feature": df.columns,
    "Data_Type": df.dtypes.astype(str),
    "Unique_Values": [df[col].nunique() for col in df.columns],
    "Missing_Values": [df[col].isnull().sum() for col in df.columns]
})

summary.to_csv("../reports/feature_summary.csv", index=False)

print("\n" + "=" * 70)
print("Feature summary saved to reports/feature_summary.csv")
print("=" * 70)