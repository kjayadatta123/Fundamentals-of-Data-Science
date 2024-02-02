import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
data = {
    'StudyTime': [2, 3, 1, 4, 5, 2, 1.5, 3.5, 4, 2.5],
    'ExamScore': [60, 70, 50, 80, 85, 65, 55, 75, 82, 68]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the dataset
print(df.head())

# Calculate the correlation coefficient
correlation_coefficient = df['StudyTime'].corr(df['ExamScore'])
print(f"\nCorrelation Coefficient: {correlation_coefficient:.2f}")

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df['StudyTime'], df['ExamScore'])
plt.title('Study Time vs. Exam Score')
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Score')
plt.show()

# Create a joint plot with regression line
sns.jointplot(x='StudyTime', y='ExamScore', data=df, kind='reg')
plt.show()

# Create a pair plot for a quick overview of relationships
sns.pairplot(df)
plt.show()

# Create a heatmap to visualize the correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
