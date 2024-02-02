import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Generate sample data (replace this with your actual data)
np.random.seed(42)  # for reproducibility
control_group = np.random.normal(loc=50, scale=10, size=50)  # replace with actual control group data
treatment_group = np.random.normal(loc=55, scale=10, size=50)  # replace with actual treatment group data

# Visualize the data using a boxplot
plt.boxplot([control_group, treatment_group], labels=['Control Group', 'Treatment Group'])
plt.title('Boxplot of Control and Treatment Groups')
plt.ylabel('Effectiveness Score')
plt.show()

# Perform an independent two-sample t-test
t_stat, p_value = stats.ttest_ind(control_group, treatment_group)

# Display the results of the hypothesis test
print(f'T-statistic: {t_stat:.4f}')
print(f'P-value: {p_value:.4f}')

# Determine statistical significance
alpha = 0.05
if p_value < alpha:
    print('The difference is statistically significant. Reject the null hypothesis.')
else:
    print('The difference is not statistically significant. Fail to reject the null hypothesis.')
