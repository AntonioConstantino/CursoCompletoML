from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

# Supongamos que tienes datos de entrada (X) y etiquetas (y)
# X tiene forma (n_muestras, n_features)
# y tiene clases codificadas como enteros: 0, 1, 2, ..., n_clases - 1

# Datos de ejemplo
X = np.random.rand(1000, 20)
y = np.random.randint(0, 4, size=(1000,))  # 4 clases

# One-hot encoding de las etiquetas
y_cat = to_categorical(y)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Definición del modelo
model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_cat.shape[1], activation='softmax'))  # Salida softmax para clasificación múltiple

# Compilación del modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
model.fit(X_train, y_train, epochs=50, batch_size=15, validation_split=0.3)

y_val=model.predict(X_test)

y_val_category=np.argmax(y_val,axis=1)
#Cuando sea 3 entonces es cosmetico y cuando sea 
#0=cosmetico ,1=electronica,2=telefonia,3=dulceria
area_liverpool=[]
for indice_y_val in y_val_category:
    if indice_y_val== 0:
        area_liverpool.append("Cosmetico")
    elif indice_y_val== 1:
        area_liverpool.append("Electronica")
    elif indice_y_val== 2:
        area_liverpool.append("Telefonia")
    elif indice_y_val== 3:
        area_liverpool.append("Dulceria")
           
# guardado ( tensorflow , h5 ,plk, joblib )