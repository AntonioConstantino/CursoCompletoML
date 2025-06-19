import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Datos dummy (relación lineal simple)
# Entrada (x) y salida (y) donde y = 2x + 1
x = np.array([0, 1, 2, 3, 4], dtype=float)
y = np.array([1, 3, 5, 7, 9], dtype=float)

# Modelo secuencial con una sola neurona (una sola capa densa)
model = Sequential([
    Dense(units=1, input_shape=[1])
])
model.add(Dense(10,activation="relu"))
model.add(Dense(1))

# Compilación del modelo: optimizador y función de pérdida
model.compile(optimizer='sgd', loss='mean_squared_error')

# Entrenamiento del modelo
model.fit(x, y, epochs=500, verbose=1)

# Prueba de predicción
print("Predicción para x=10:", model.predict(np.array([10.0])))#[0][0])