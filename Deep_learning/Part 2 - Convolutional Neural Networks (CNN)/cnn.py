#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Antonio Constantino
"""

# Redes Neuronales Convolucionales

# Instalar Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Instalar Tensorflow y Keras
# conda install -c conda-forge keras

# Cargar el modelo

#from keras.models import load_model
#classifier=load_model(r"classifier")


# Parte 1 - Construir el modelo de CNN

# Importar las liobrerías y paquetes
from keras.models import Sequential #iniciar la red neuronal con pesos aleatorios 
from keras.layers import Conv2D  # Crear capa de convolución (Detectar caracteristicas )
from keras.layers import MaxPooling2D # Función de pooling 
from keras.layers import Flatten # Aplanado 
from keras.layers import Dense #las conexiones entre capas 

# Inicializar la CNN
classifier = Sequential()

# Paso 1 - Convolución 
# Detectores de razgos ( con 32 todo es perro , con 50 predice bien )

classifier.add(Conv2D(filters = 50,# Filtros ( las matrices de razgos )
                      kernel_size = (3, 3), # tamaño del kernel (es a color o es a blanco y negro)
                      input_shape = (64, 64, 3), #Tamaño de entrada imagenes ( alto, ancho y canal de color )
                      activation = "relu"))# Rectificador Lienal Unitario ( eliminar valores negativos )
# escala de color es 1( blanco y negro ), 3 RGB (3 colores )


# Paso 2 - Max Pooling
# tomamos la ventana de los valores despues de la convolución 
# Se van haciendo chiquita la matriz de convolución 
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Una segunda capa de convolución y max pooling
classifier.add(Conv2D(filters = 32,kernel_size = (3, 3), activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2,2)))

# Paso 3 - Flattening
# Convierte en las entradas de la Red neuronal Artificial
# lista de caracteristicas 
classifier.add(Flatten())

# Paso 4 - Full Connection ( totalmente conectada)
#Dense Añade capas ocultas 
classifier.add(Dense(units = 128,# numero de perceptrones
                     activation = "relu")) # Se despierta la neurona o no
# Tomar el promedio de nodos de netrada y final ( units)
classifier.add(Dense(units = 1,# capa de salida
                     activation = "sigmoid")) # La probabilidad que sea un dato u el otro 

# Compilar la CNN
classifier.compile(optimizer = "adam", # algortimo estocastico 
                   loss = "binary_crossentropy", # Perdida ( binaria or que entrenamos entre perros y gatos )
                   metrics = ["accuracy"])# Extactitud


# Parte 2 - Ajustar la CNN a las imágenes para entrenar 
from keras.preprocessing.image import ImageDataGenerator

# 
train_datagen = ImageDataGenerator(
        rescale=1./255,# convierte de rgb a un decimal ( el rango de valores es pequeño)
        shear_range=0.2,
        zoom_range=0.2,# porcentajede zoom
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)# reesaclamos 

# se mete a buscar las imagenes y las carga en un objeto con todas las imagenes 
training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),# tiene qu tener el mismo tamaño decalrado arriba 
                                                batch_size=32,# de 32 en 32 imagenes ( para correjir los pesos )
                                                class_mode='binary')
#Este manda un warning y no temina de entrenar 
#classifier.fit(training_dataset,# conjunto de Training
#                        steps_per_epoch=800,# puede ser el numero total de imagenes 
#                        epochs=25, #Numero de Iteraciones 
#                        validation_data=testing_dataset,# conjunto de validación 
#                        validation_steps=2000) # numero de pasos 

# Este es lo mas correcto de los entrenamientos de la red neuronal  (tarda unas 2 horas aprox)
classifier.fit_generator(
   training_dataset, #conjunto de entrenamiento
   steps_per_epoch = training_dataset.n//32, #Muestras que toma en cada ciclo de entrenamiento, pasaremos todas las imagenes
   epochs=25, #Cuantas epocas usaremos para entrenar
   validation_data=testing_dataset, #Conjunto de validación
   validation_steps=2000) #Cada cuantos pasadas validaremos nuestro resultado en este caso 2 cada 8 epocas



# Parte 3 - Cómo hacer nuevas predicciones
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

test_image = load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0) 
result = classifier.predict(test_image)
training_dataset.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

#guarda el modelo (predice muy bien)
#classifier.save(r"classifier")

