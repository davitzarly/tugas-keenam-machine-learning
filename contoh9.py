# Contoh Implementasi dengan Scikit-learn 
import numpy as np 
from sklearn.datasets import make_classification 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

# Membuat data sampel 
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, 
                           n_redundant=5, random_state=42) 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                  random_state=42) 

# Membangun model dengan early stopping 
model = GradientBoostingClassifier(n_estimators=1000,
                                   validation_fraction=0.2,
                                   n_iter_no_change=10,
                                   learning_rate=0.01)

# Melatih model 
model.fit(X_train, y_train) 

# Prediksi dan evaluasi 
y_pred = model.predict(X_val) 
accuracy = accuracy_score(y_val, y_pred) 
print(f'Validation Accuracy: {accuracy:.4f}')
