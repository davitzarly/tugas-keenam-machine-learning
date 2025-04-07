import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Buat data normal
data_normal = np.random.normal(loc=0, scale=1, size=1000)

# Plot boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=data_normal)
plt.title('Boxplot of Normal Distribution')
plt.xlabel('Values')
plt.show()
