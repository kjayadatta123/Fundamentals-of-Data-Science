import scipy.stats as stats
import numpy as np

# Sample data for the drug group
data_drug = np.array([3, 5, 2, 6, 4, 8, 1, 7, 3, 5, 2, 6, 4, 8, 1, 7, 3, 5, 2, 6, 4, 8, 1, 7, 3, 5, 2, 6, 4, 8, 1, 7, 3, 5, 2, 6, 4, 8, 1, 7, 3, 5, 2, 6, 4, 8, 1, 7])

# Sample data for the placebo group
data_placebo = np.array([2, 4, 1, 5, 3, 7, 2, 6, 4, 8, 1, 5, 3, 7, 2, 6, 4, 8, 1, 5, 3, 7, 2, 6, 4, 8, 1, 5, 3, 7, 2, 6, 4, 8, 1, 5, 3, 7, 2, 6, 4, 8, 1, 5, 3, 7, 2, 6, 4, 8])

# Function to calculate confidence interval
def calculate_confidence_interval(data):
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # using ddof=1 for sample standard deviation
    sample_size = len(data)
    
    critical_value = stats.norm.ppf(0.975)  # For a 95% confidence interval (two-tailed)
    margin_of_error = critical_value * (std_dev / np.sqrt(sample_size))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    
    return confidence_interval

# Calculate confidence intervals
confidence_interval_drug = calculate_confidence_interval(data_drug)
confidence_interval_placebo = calculate_confidence_interval(data_placebo)

# Print the results
print("Sample Data for Drug Group:", data_drug)
print("95% Confidence Interval for Drug Group:", confidence_interval_drug)

print("\nSample Data for Placebo Group:", data_placebo)
print("95% Confidence Interval for Placebo Group:", confidence_interval_placebo)
