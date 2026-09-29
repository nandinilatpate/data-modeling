import numpy as np
from scipy.stats import ttest_1samp

# Sample data
sample = [52, 48, 51, 49, 50, 53, 47, 51, 50, 49]

# Known population mean
population_mean = 50

# Perform hypothesis test
t_stat, p_value = ttest_1samp(sample, population_mean)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject the Null Hypothesis")
else:
    print("Fail to Reject the Null Hypothesis")