
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the existing dataset
df = pd.read_csv("student.csv")

# Create a folder to save graphs
os.makedirs("charts", exist_ok=True)

sns.set_theme(style="whitegrid")

# 1. Bar chart: Average marks by gender
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="gender", y="marks")
plt.title("Average Marks by Gender")
plt.xlabel("Gender")
plt.ylabel("Average Marks")
plt.tight_layout()
plt.savefig("charts/bar_chart.png")
plt.show()

# 2. Histogram: Distribution of marks
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="marks", bins=5, kde=True)
plt.title("Distribution of Student Marks")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("charts/histogram.png")
plt.show()

# 3. Scatter plot: Study hours vs marks
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df, x="study_hours", y="marks", s=80
)
plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("charts/scatter_plot.png")
plt.show()

# 4. Line chart: Marks by student
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x="student_id", y="marks", marker="o")
plt.title("Marks by Student")
plt.xlabel("Student ID")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("charts/line_chart.png")
plt.show()

# 5. Heatmap: Correlation between numerical variables
numeric_data = df[
    ["study_hours", "attendance", "marks"]
]
correlation = numeric_data.corr()

plt.figure(figsize=(7, 5))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("charts/heatmap.png")
plt.show()

# Print basic findings
print("\nDATA VISUALIZATION FINDINGS")
print("---------------------------")
print("Average marks:", round(df["marks"].mean(), 2))
print("Highest marks:", df["marks"].max())
print("Lowest marks:", df["marks"].min())
print(
    "Correlation between study hours and marks:",
    round(df["study_hours"].corr(df["marks"]), 2)
)

print("\nAll five charts saved in the charts folder.")
