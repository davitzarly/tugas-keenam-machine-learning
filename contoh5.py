import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
# Membuat data sampel 
data = { 
'Age': [25, 30, 35, 40, 100, 45, 50, 25, 30, 35, 40, 45, 50, 150], 
'Income': [50000, 80000, 60000, 70000, 200000, 75000, 80000, 50000, 80000, 
60000, 70000, 75000, 80000, 300000] 
} 
# Membuat dataframe 
df = pd.DataFrame(data) 
# Menampilkan data 
print(df) 
# Visualisasi Boxplot 
plt.figure(figsize=(10, 6)) 
sns.boxplot(x=df['Age']) 
plt.title('Boxplot Age') 
plt.xlabel('Age') 
plt.show()
# Visualisasi Scatter Plot 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='Age', y='Income', data=df) 
plt.title('Scatter Plot Age vs Income') 
plt.xlabel('Age') 
plt.ylabel('Income') 
plt.show() 
 
# Identifikasi Outliers dengan IQR 
Q1 = df['Age'].quantile(0.25) 
Q3 = df['Age'].quantile(0.75) 
IQR = Q3 - Q1 
 
outliers_IQR = df[(df['Age'] < (Q1 - 1.5 * IQR)) | (df['Age'] > (Q3 + 1.5 * IQR))] 
print("\nOutliers berdasarkan IQR:") 
print(outliers_IQR) 
 
# Identifikasi Outliers dengan Z-Score 
from scipy import stats 
 
z_scores = np.abs(stats.zscore(df['Age'])) 
outliers_z = df[(z_scores > 3)] 
print("\nOutliers berdasarkan Z-Score:") 
print(outliers_z) 