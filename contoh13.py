# Droout dengan Keras 
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
 
# Membuat data sampel 
X_train = np.random.rand(1000, 20) 
y_train = np.random.randint(2, size=(1000, 1)) 
X_val = np.random.rand(200, 20) 
y_val = np.random.randint(2, size=(200, 1)) 
 
# Membangun model dengan Dropout 
model = Sequential() 
model.add(Dense(64, activation='relu', input_dim=20)) 
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1, activation='sigmoid')) 
 
# Kompilasi model 
model.compile(optimizer='adam', loss='binary_crossentropy', 
metrics=['accuracy']) 
 
# Melatih model 
history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
validation_data=(X_val, y_val)) 
 
# Menampilkan ringkasan model 
print(model.summary())