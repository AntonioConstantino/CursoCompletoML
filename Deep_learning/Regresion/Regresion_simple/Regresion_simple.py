import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Datos sintéticos
X = np.random.rand(100) * 100
y = 3.5 * X + np.random.randn(100) * 10

# Modelo
model = keras.Sequential([
    layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Callback para registrar historial

historial = model.fit(X, 
                      y, 
                      epochs=500,#50, 100, 250, 500
                      batch_size=15,
                      validation_split=0.2, 
                      verbose=1)


# Gráfica de pérdida
plt.plot(historial.history['loss'], label='Entrenamiento')
plt.plot(historial.history['val_loss'], label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()