# Redes Neuronales Recurrentes (RNR)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 19:32:12 2020

@author: Antonio Constantino
"""
# Parte 1 - Preprocesado de los datos

# Importación de las librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importar el dataset de entrenamiento
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set  = dataset_train.iloc[:, 1:2].values# todas las filas , solo una columna 

# Escalado de características
from sklearn.preprocessing import MinMaxScaler # normalizacion 
# Generamos la funcion para normalizar 
#xnorm= (x-min(x))/(max(x)-min(x))
sc = MinMaxScaler(feature_range = (0, 1))# Dimencionamos entre 0 y 1
training_set_scaled = sc.fit_transform(training_set)# aplicamos la funcion 

# Crear una estructura de datos con 60 timesteps( 60 dias) y 1 salida
X_train = [] 
y_train = []
# dale 59 dias y prediece el dia 60
# del 61 dias y predice el dia 121, etc 
for i in range(60, len(dataset_train)):
    # separa los el dataset esclado de test y training desde los indices 
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

#convertimos en un array
X_train, y_train = np.array(X_train), np.array(y_train)

# Redimensión de los datos para rnn (bach_size, timestep,input_dim) asi lo pide keras 
X_train = np.reshape(X_train,# este objeto
                     (X_train.shape[0], X_train.shape[1], 1))#numero de filas ,numero de columnas , dimencion 1

# Parte 2 - Construcción de la RNR ( Red neuronal recurrente )
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Dropout

# Inicialización del modelo
regressor = Sequential()

# Añadir la primera capa de LSTM(Long Short Temp Memory) y la regulariación por Dropout
regressor.add(LSTM(units = 50, # Numero de neuronas 
                   return_sequences = True, #Si van a recalcularse los pesos
                   input_shape = (X_train.shape[1], 1) ))# tamaño del df
regressor.add(Dropout(0.2))# Evitar el overfiting 

# Añadir la segunda capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, 
                   return_sequences = True ))# Permite pesos hacia atras
regressor.add(Dropout(0.2))

# Añadir la tercera capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

# Añadir la cuarta capa de LSTM y la regulariación por Dropout
regressor.add(LSTM(units = 50))# return esta en false po default
regressor.add(Dropout(0.2))

# Añadir la capa de salida
regressor.add(Dense(units = 1))# numero de salidas 

# Compilar la RNR
regressor.compile(optimizer = 'adam', 
                  loss = 'mean_squared_error')# Con base al error vemos que tan bien  hace la regresion 

# Ajustar la RNR al conjunto de entrenamiento
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Parte 3 - Ajustar las predicciones y visualizar los resultados

# Obtener el valor de las acciones reales  de Enero de 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')# cargamos los datos 
real_stock_price = dataset_test.iloc[:, 1:2].values#

# Obtener la predicción de la acción con la RNR para Enero de 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values# todos los valores 
inputs = inputs.reshape(-1,1)# lo redimencionamos 
inputs = sc.transform(inputs)# los escalamos 
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])#
X_test = np.array(X_test) # convertimos en array
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))# rediemencionamos 

predicted_stock_price = regressor.predict(X_test)# calculamos la prediccion 
#sc es el transformador de normalizacion
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # convertimos de escalado a los datos normales 

# Visualizar los Resultados
plt.plot(real_stock_price, color = 'red', label = 'Precio Real de la Accion de Google')
plt.plot(predicted_stock_price, color = 'blue', label = 'Precio Predicho de la Accion de Google')
plt.title("Prediccion con una RNR del valor de las acciones de Google")
plt.xlabel("Fecha")
plt.ylabel("Precio de la accion de Google")
plt.legend()
plt.show()

#calcular el error cuadrado medio 
import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))# ver en variables 

#Guardar modelo
regressor.save(r"regressor")

# cargar un modelo entrenado 
#from keras.models import load_model
#regressor=load_model("regressor")