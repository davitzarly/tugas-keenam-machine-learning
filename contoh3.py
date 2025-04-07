import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Membuat data distribusi normal
data_normal = np.random.normal(loc=0, scale=1, size=1000)

# Plot density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data_normal, shade=True)
plt.title('Density Plot of Normal Distribution')
plt.xlabel('Values')
plt.ylabel('Density')
plt.show()
