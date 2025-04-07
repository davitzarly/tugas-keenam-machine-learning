# Regularisasi dengan L2 (Ridge Regression) di Scikit-learn 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import Ridge 
from sklearn.metrics import mean_squared_error 
# Membuat data sampel 
X = np.random.rand(100, 5) 
y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(100) * 0.1 
# Split data menjadi train dan test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
# Melatih model dengan Ridge Regression 
ridge_reg = Ridge(alpha=1.0) 
ridge_reg.fit(X_train, y_train)
# Prediksi dan evaluasi 
y_pred_train = ridge_reg.predict(X_train) 
y_pred_test = ridge_reg.predict(X_test) 
 
print("Train MSE:", mean_squared_error(y_train, y_pred_train)) 
print("Test MSE:", mean_squared_error(y_test, y_pred_test)) 