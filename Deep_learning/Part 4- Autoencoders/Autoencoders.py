import numpy as np # calculo numerico 
from tensorflow.keras.datasets import mnist # Imagenes de numeros escritos a mano 
from tensorflow.keras.models import Model # Modelos 
from tensorflow.keras.layers import Input, Dense # capas 
import matplotlib.pyplot as plt # graficar 

# 1. Cargar los datos
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# 2. Definir la arquitectura
input_dim = x_train.shape[1]
encoding_dim = 32  # Compresión a 32 dimensiones

input_img = Input(shape=(input_dim,))
## Esta es otra forma de crear aquitecturas de redes neuronales 

encoded = Dense(encoding_dim, activation='relu')(input_img)# Grande inicial
decoded = Dense(input_dim, activation='sigmoid')(encoded)# chiquita que aprende las caracteristicas 

autoencoder = Model(input_img,# de que tamaño resibo imagnes 
                    decoded)# Como l¿necesito que creees el Decoder ( salida)

# 3. Compilar y entrenar
autoencoder.compile(optimizer='adam', # adam 
                    loss='binary_crossentropy') # 0 y 1

autoencoder.fit(x_train, x_train,# Datos de entrenamiento
                epochs=20,# 
                batch_size=256,
                shuffle=True,# deslizarme 
                validation_data=(x_test, x_test))

# 4. Visualizar resultados
encoded_imgs = autoencoder.predict(x_test) # ya tienes un modelo de IA entrenado y predices nuevas images 

n = 10  # Número de imágenes a mostrar
plt.figure(figsize=(20, 4))
for i in range(n):
    # Imagen original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    # Imagen reconstruida
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()