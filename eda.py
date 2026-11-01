import pandas as pd

# Load the dataset
df = pd.read_csv("student.csv")

# Display the first five rows
print(df.head())

# Display the number of rows and columns
print(df.shape)

# Display column names and data types
print(df.info())

# Check for missing values
print(df.isnull().sum())

# Generate statistical information
print(df.describe())

# Check for duplicate rows
print(df.duplicated().sum())
from scipy.stats import pearsonr

correlation, p_value = pearsonr(
    df["study_hours"],
    df["marks"]
)

print("\nHypothesis Test: Study Hours vs Marks")
print("Correlation:", round(correlation, 3))
print("P-value:", round(p_value, 4))

if p_value < 0.05 and correlation > 0:
    print("Result: Evidence of a positive relationship.")
else:
    print("Result: Insufficient evidence of a positive relationship.")