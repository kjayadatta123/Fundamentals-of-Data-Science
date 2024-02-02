import scipy.stats as stats
import numpy as np

# Replace these with your actual data
conversion_rates_A = np.array([0.1, 0.15, 0.2, 0.18, 0.25])  # Example data for website design A
conversion_rates_B = np.array([0.08, 0.12, 0.19, 0.22, 0.21])  # Example data for website design B

# Perform independent samples t-test
t_statistic, p_value = stats.ttest_ind(conversion_rates_A, conversion_rates_B)

# Check the p-value
alpha = 0.05
print("p-value:", p_value)

# Make a decision based on the p-value
if p_value < alpha:
    print("There is a statistically significant difference in mean conversion rates between website design A and B.")
else:
    print("There is no statistically significant difference in mean conversion rates between website design A and B.")
