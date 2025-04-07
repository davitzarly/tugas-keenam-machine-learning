import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 
# Generate data from a normal distribution 
data_normal = np.random.normal(loc=0, scale=1, size=1000) 
# Plot histogram 
plt.figure(figsize=(10, 6)) 
sns.histplot(data_normal, bins=30, kde=True) 
plt.title('Histogram of Normal Distribution') 
plt.xlabel('Values') 
plt.ylabel('Frequency') 
plt.show() 