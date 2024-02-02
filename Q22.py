import pandas as pd
from scipy import stats

# Load the customer reviews data from the CSV file
file_path = "customer_reviews.csv"
df = pd.read_csv("/Users/lakshminarayanamandi/Downloads/Movies/FODS/data.csv")

# Display the first few rows of the dataframe to understand the data
print(df.head())

# Calculate the mean and standard deviation of the ratings
mean_rating = df['Rating'].mean()
std_dev = df['Rating'].std()

# Calculate the sample size
sample_size = len(df)

# Set the confidence level (e.g., 95% confidence interval)
confidence_level = 0.95

# Calculate the standard error of the mean
std_error = std_dev / (sample_size ** 0.5)

# Calculate the margin of error
margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * std_error

# Calculate the confidence interval
confidence_interval = (mean_rating - margin_of_error, mean_rating + margin_of_error)

# Display the results
print(f"Mean Rating: {mean_rating:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Sample Size: {sample_size}")
print(f"Confidence Interval ({confidence_level * 100}%): {confidence_interval}")
