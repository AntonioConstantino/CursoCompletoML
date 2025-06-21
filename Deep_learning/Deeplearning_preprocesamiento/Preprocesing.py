import pandas as pd #bases de datos
import numpy as np # procesar y operaciones matematicas muy rapidas 
import seaborn # graficar 
from sklearn.impute import SimpleImputer #Imputar valores 
from sklearn.model_selection import train_test_split # separar conunto de train y test
from sklearn.preprocessing import OrdinalEncoder # convertit variables categoricas a numeros
from sklearn.preprocessing import StandardScaler # escalar los valores
from sklearn.metrics import confusion_matrix # evaluar el modelo
from tensorflow import keras # Crea el modelo de IA 
from tensorflow.keras.models import Sequential # Apila las capas de la red neuraonal
from tensorflow.keras.layers import Dense# tipo de capa Full conect 



titanic=seaborn.load_dataset('titanic')#Cargamos el dataframe 
# como borro la columna que no cumple con este criterio 
titanic.drop(columns="deck",inplace=True)#eliminamos la columna
#imputamos  las columnas que tienen campos vacios
titanic['age'].fillna(titanic['age'].mean(), inplace=True) # con la media por que es numerico
titanic["embarked"].fillna(titanic["embarked"].mode(),inplace=True) # Con la moda por que es categorico
titanic["embark_town"].fillna(titanic["embarked"].mode(),inplace=True)
titanic["alone"]=np.where( titanic["alone"],1,0)
titanic["adult_male"]=np.where(titanic["adult_male"],1,0)
#Eliminamos las filas que tienen valores vacios 
titanic.dropna(inplace=True)
print(titanic.isna().sum())
copletitud=titanic.isna().sum()

#Similar a one hot encoder
encoder = OrdinalEncoder() #a,a,b,c -> 0,0,1,2 Tipo de conversion 
columnas_transformadas_ordinales = encoder.fit_transform(titanic.select_dtypes(include=["object", "category"]).values)# aplico el metodo a 2 columnas es especifico .selecvalues 

titanic[titanic.select_dtypes(include=["object", "category"]).columns]=columnas_transformadas_ordinales

#defino el metodo
sc=StandardScaler()
#Entreno para escalar para variables numericas
sc=sc.fit(titanic[["age","fare"]])
x_sc_scaled=sc.transform(titanic[["age","fare"]])
titanic[["age","fare"]]=x_sc_scaled

#Vamos a obtener las posibles variables para predecir un resultado
x=titanic.drop(columns="alive")
#separo la(s) columnas para predecir 
y=titanic["alive"]


#Separamos para entrenamiento y validacion 
X_train, X_test, y_train, y_test = train_test_split(x,#Variables a las cuales ya les aplique el preprocesado
                                                     y, #la columna que tiene las repuesta (Si sobrevivio)
                                                     test_size=0.20,# Esto es de evaluacion (y es el 20% , entreno con el 80%) 
                                                     random_state=1996)#Para reproducibilidad del experiemnto ( Para que nos de los mismo resultados  )


# entrenamos un modelo de ia de tipo keras 

model=Sequential()
# agregamos una capa entrada 
model.add(
    Dense( units=X_train.shape[1] ,#numero de columnas ( variables predictoras )
                input_shape=[X_train.shape[1]],
                activation="relu"#https://keras.io/api/layers/activations/
    )
    )
# capas ocultas 
#agregar capa 

#Agregamos capa de salida
model.add(Dense(1,
                activation="sigmoid")) # Sigmoide solo da valores entre 0 y 1

# Compilación del modelo: optimizador y función de pérdida
model.compile(optimizer='adam', #sgd https://keras.io/api/optimizers/
              loss='binary_crossentropy', #https://keras.io/api/losses/
              metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.FalseNegatives(),
        ]
              )

model.fit(X_train ,# variables predictoras
          y_train,# Variable objetivo
          epochs=50, # cuantas veces va a intentar encontrar el error minimo (clasificacion regresion )
        batch_size=25, # tamaño de los saltos 
          verbose=1)# 1 Muestre como va el modelo y 0 (cero ) no me muestre naa 

# predecir ( ojo : deben ser la misma cantidad de variables que puse en x)

y_pred=model.predict(X_test)
pd.Series(y_pred.ravel()).to_csv("predicciones.csv")
y_series=pd.Series(y_pred.ravel())
# estos son numeros decimales (probabilidades )
#necesitamos meter humbrales 
sobrevivientes90porciento=y_pred>.90
sobrevivientes_1_o_0=np.where(sobrevivientes90porciento,1,0)

np.save("./sobrevivientes.npy", sobrevivientes_1_o_0)
# para cargarlo loaded_data = np.load("./sobrevivientes.npy")


