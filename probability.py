import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data
data = np.random.normal(50, 10, 1000)

# Histogram
plt.hist(data, bins=30, density=True, alpha=0.6)

# PDF
x = np.linspace(min(data), max(data), 100)
pdf = norm.pdf(x, np.mean(data), np.std(data))

plt.plot(x, pdf)
plt.xlabel("Values")
plt.ylabel("Probability Density")
plt.title("Probability Distribution")
plt.show()