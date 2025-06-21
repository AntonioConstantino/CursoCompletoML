# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 21:57:27 2025
ejemplo de clasificacion binaria
@author: jacj2
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense# Full conect 
import  seaborn
# Data 
#conjunto de entrenamiento y validacion 
#preporocesamiento 
# esclaado #dummi #One hot encoding #compuetitud # imputar valore , quitar faltantes 


model=Sequential()
# geragamos una capa entrada 
model.add(
    Dense( units=3 ,#numero de columnas ( variables predictoras )
                input_shape=[3],
                activation="relu"#https://keras.io/api/layers/activations/
    )
    )
# capas ocultas 
#agregar capa 

#Agregamos capa de salida
model.add(Dense(1,
                activation="sigmoid"))

# Compilación del modelo: optimizador y función de pérdida
model.compile(optimizer='sgd', # https://keras.io/api/optimizers/
              loss='MeanAbsolutePercentageError') #https://keras.io/api/losses/

model.fit(x_train ,# variables predictoras
          y_train,# Variable objetivo
          epochs=500, # cuantas veces va a intentar encontrar el error minimo (clasificacion regresion )
          verbose=1)# 1 Muestre como va el modelo y 0 (cero ) no me muestre naa 

# predecir ( ojo : deben ser la misma cantidad de variables que puse en x)
model.predict()
# se meten los valores esclados ( mismo esclador que entrene ): esclador tengo que guardarlo 
print("Predicción para x=10:", model.predict(np.array([10.0,20,80])))#[0][0])

