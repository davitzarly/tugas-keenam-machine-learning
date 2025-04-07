import pandas as pd 
import numpy as np 
 
# Membuat data sampel dengan missing values 
data = { 
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'], 
    'Age': [25, np.nan, 35, 45, np.nan], 
    'Income': [50000, 60000, np.nan, 80000, 90000], 
    'Gender': ['F', 'M', np.nan, 'M', 'F'] 
} 
 
# Membuat dataframe 
df = pd.DataFrame(data) 
 
# Menampilkan data 
print("Original DataFrame:") 
print(df) 
 
# Menghapus baris dengan missing values 
df_dropna = df.dropna() 
print("\nDataFrame setelah menghapus baris dengan missing values:") 
print(df_dropna) 
 
# Mengisi missing values dengan mean (untuk kolom numerik) 
df['Age'].fillna(df['Age'].mean(), inplace=True) 
df['Income'].fillna(df['Income'].mean(), inplace=True) 
print("\nDataFrame setelah mengisi missing values dengan mean:") 
print(df) 
 
# Mengisi missing values dengan mode (untuk kolom kategorikal) 
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True) 
print("\nDataFrame setelah mengisi missing values dengan mode:") 
print(df) 
 
# Mengisi missing values dengan metode forward fill 
df_ffill = df.fillna(method='ffill') 
print("\nDataFrame setelah mengisi missing values dengan forward fill:")
print(df_ffill) 
 
# Mengisi missing values dengan metode backward fill 
df_bfill = df.fillna(method='bfill') 
print("\nDataFrame setelah mengisi missing values dengan backward fill:") 
print(df_bfill) 
 
# Mengisi missing values dengan interpolasi 
df_interpolate = df.interpolate(method='linear') 
print("\nDataFrame setelah mengisi missing values dengan interpolasi:") 
print(df_interpolate)