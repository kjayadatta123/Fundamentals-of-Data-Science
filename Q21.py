import numpy as np
import pandas as pd
import scipy.stats as stats

# Load data from the CSV file
try:
    data = pd.read_csv("/Users/lakshminarayanamandi/Downloads/Movies/FODS/data.csv")
except FileNotFoundError:
    print("Error: 'rare_elements.csv' file not found.")
    exit()

# Function to calculate confidence interval
def calculate_confidence_interval(data, confidence_level):
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # using ddof=1 for sample standard deviation
    sample_size = len(data)

    # Calculate critical value based on confidence level
    critical_value = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)

    # Calculate margin of error
    margin_of_error = critical_value * (sample_std / np.sqrt(sample_size))

    # Calculate confidence interval
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

    return confidence_interval

# User inputs
sample_size = int(input("Enter the sample size: "))
confidence_level = float(input("Enter the confidence level (e.g., 0.95 for 95%): "))
precision = float(input("Enter the desired level of precision: "))

# Check if the sample size is less than the total data size
if sample_size >= len(data):
    print("Error: Sample size should be less than the total data size.")
    exit()

# Randomly select a sample from the data
random_sample = np.random.choice(data['Concentration'], size=sample_size, replace=False)

# Calculate confidence interval
confidence_interval = calculate_confidence_interval(random_sample, confidence_level)

# Display the results
print("\nResults:")
print(f"Sample Mean: {np.mean(random_sample)}")
print(f"Confidence Interval: {confidence_interval}")
