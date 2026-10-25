import pandas as pd

# Load dataset
df = pd.read_csv("../dataset/data.csv", sep=";")

print("=" * 60)
print("STUDENT DROPOUT DATASET ANALYSIS")
print("=" * 60)

# Dataset size
print("\n1. Dataset Shape:")
print(df.shape)

# Column names
print("\n2. Total Columns:")
print(len(df.columns))

print("\n3. Column Names:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# First 5 rows
print("\n4. First 5 Records:")
print(df.head())

# Data types
print("\n5. Data Types:")
print(df.dtypes)

# Missing values
print("\n6. Missing Values:")
print(df.isnull().sum())

# Duplicate records
print("\n7. Duplicate Records:")
print(df.duplicated().sum())

# Target distribution
print("\n8. Target Distribution:")
print(df["Target"].value_counts())

print("\n9. Target Percentage:")
print(df["Target"].value_counts(normalize=True) * 100)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)