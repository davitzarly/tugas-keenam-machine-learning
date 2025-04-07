import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
 
# Membaca dataset 
df = pd.read_csv('data.csv') 
 
# Melihat struktur data 
print("Shape of the dataset:", df.shape) 
print("\nInfo about the dataset:") 
print(df.info()) 
 
# Melihat data sekilas 
print("\nFirst few rows of the dataset:") 
print(df.head()) 
print("\nStatistical summary of the dataset:") 
print(df.describe()) 
 
# Visualisasi distribusi 
plt.figure(figsize=(10, 6)) 
sns.histplot(df['Age'], bins=30, kde=True) 
plt.title('Distribusi Age') 
plt.xlabel('Age') 
plt.ylabel('Frequency') 
plt.show() 
 
# Visualisasi hubungan 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='Age', y='Income', data=df) 
plt.title('Hubungan Age dan Income') 
plt.xlabel('Age') 
plt.ylabel('Income') 
plt.show() 
 
# Visualisasi kategori 
plt.figure(figsize=(10, 6)) 
sns.countplot(x='Gender', data=df) 
plt.title('Distribusi Gender') 
plt.xlabel('Gender') 
plt.ylabel('Count') 
plt.show() 
 
# Analisis statistik 
print("\nCorrelation matrix:") 
print(df.corr()) 
 
# Mengidentifikasi missing values 
print("\nMissing values in each column:") 
print(df.isnull().sum()) 
 
# Mengimputasi missing values dengan mean 
df['Age'].fillna(df['Age'].mean(), inplace=True) 
 
# Mengidentifikasi outliers dengan boxplot 
plt.figure(figsize=(10, 6)) 
sns.boxplot(x=df['Age']) 
plt.title('Boxplot Age') 
plt.xlabel('Age') 
plt.show() 
 
# Normalisasi fitur numerik 
from sklearn.preprocessing import StandardScaler 
 
scaler = StandardScaler() 
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']]) 
 
# Membuat fitur baru 
df['Age_Income_Ratio'] = df['Age'] / df['Income'] 
 
# Melihat data setelah preprocessing 
print("\nFirst few rows of the dataset after preprocessing:") 
print(df.head()) 
