#!/usr/bin/env python
# coding: utf-8

# # Analisis de completitud (ok)
# # Rellenar los vacios ( imputacion )(ok)
# * Numericos (ok)
# * Categoricos (en el lenguaje R puedo usar el paquete mice)(ok)
# 
# # variables predictoras y a predecir (ok)
# ### aprendizaje supervisado (etiquetas) y no supervisado (no tengo las etiquetas),# aprendizaje semi supervisado ,aprendizaje por refuerzo
# # Escalar los valores (minmax scaler , standarscaler) (ok)
# # pca ( MODULOS AVANZADOS ) ( DE LA VARIANZA ME DICE QUE COLUMNAS ME SIRVEN Y CUALES NO )
# # Si tengo variables categoricas (Dummy o onehot_encoding )(ok)
# # Conjunto de test y de train (prueba y de entrenamiento )(ok)
# # seleccionar y entrenar un modelo de IA 
# # Evaluar las caracteristica 
# # metricas ( Curva ROC , AUC, Matris de confusion )
# 
# # Bonuss ( pipelines) Popote

# In[67]:


import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
import seaborn # graficar 


# In[123]:


# creo una copia
df=seaborn.load_dataset('titanic')


# In[68]:


# cargo los datos 
titanic=seaborn.load_dataset('titanic')


# In[69]:


# guardo los datos en csv , y se cre un archivo .bi
titanic.to_csv(r"data/df_titanic.csv")


# In[70]:


# vemos el numero de filas y columnas 
titanic.shape


# In[71]:


# Porcentaje por columna vacia 
# si yo tengo arriba del 40 % vacio ( me conviene borrarla ) # si hago esto , estaria sesgando el modelo 
titanic.isna().sum()/titanic.shape[0]


# In[72]:


titanic.embarked


# In[73]:


# esto es un ejemplo de como borrar varias columnas con una instruccion 
#titanic.drop(columns=["deck","who"])


# In[74]:


# como borro la columna que no cumple con este criterio 
titanic.drop(columns="deck",inplace=True)


# In[75]:


titanic.columns


# In[76]:


titanic.age


# In[77]:


# Que significa NAN
#Not Number Avalable ( sin valor disponible )


# In[78]:


# Que hace la validacion de isna( da una lista de boleanos )
titanic.age.isna()


# # Imputacion valores numericos 

# In[79]:


#como rellenar con la media 
titanic['age'].fillna(titanic['age'].mean(), inplace=True)


# # Eliminar las filas 

# In[80]:


# Eliminar todas filas vacias 177, 
#titanic.dropna(inplace=True)


# # Como imputar valores categoricos

# In[81]:


#llena los valores faltantes con la moda , el valor que mas se repite
titanic["embarked"].fillna(titanic["embarked"].mode(),inplace=True)


# In[82]:


# como imputar valores categoricos

titanic['age'].fillna(titanic['age'].mean(), inplace=True)


# # Solucion de simple imputer 

# In[83]:


True==1


# In[84]:


titanic.info()


# In[85]:


#for 
#np.where 
lista_numeros_number_adult_male=[]
for i in titanic.adult_male:
    if i==True:
        #print(f"Entro el true: {i}")
        lista_numeros_number_adult_male.append(1)
    elif i==False:
        #print(f"Entro el false: {i}")
        lista_numeros_number_adult_male.append(0)


# In[86]:


def conversor_bool_to_int(nombre_columna):
    lista_interna =[]
    for i in titanic[nombre_columna]:
        if i==True:
            lista_interna.append(1)
        elif i==False:
            lista_interna.append(0)
    return lista_interna


# In[87]:


titanic["alone"]=np.where( titanic["alone"],1,0)
titanic["adult_male"]=np.where(titanic["adult_male"],1,0)


# In[88]:


titanic.info()


# In[ ]:





# In[89]:


titanic.select_dtypes("number")


# In[90]:


imp_mean = SimpleImputer(missing_values=np.nan, # tipo de valor faltante
                         strategy='mean')# Media 
#sobre escribo en las columnas lo el array de valores numericos
titanic[titanic.select_dtypes("number").columns]=imp_mean.fit_transform(titanic.select_dtypes("number")) # mas de una columna 

# Mencionar que ehora el objeto es un aray de numpy, necesito recordar como operar numpy ( a veces no es tan facil)


# # Como seleccionar solo los numericos de los dataframe 

# In[91]:


titanic.select_dtypes("number")


# In[92]:


titanic.select_dtypes("object")


# # dummy vs one Hot encoder

# In[93]:


titanic["sex"]


# In[94]:


#Que es un valor dummy 
pd.get_dummies(titanic["sex"])


# In[95]:


pd.get_dummies(titanic.embarked)


# # diferencias entre atributos de un modulo/clase de sklearn 
# * fit (solo entrena)
# * fit_transform ( entrena y transforma )
# * transform (solo transforma , y ya debe estar entrenado)

# In[96]:


titanic.select_dtypes("object")


# In[97]:


titanic["class"]


# In[98]:


#convertimos en array 
titanic.select_dtypes("object").values


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
encoder=encoder.fit(titanic.select_dtypes(include=["object", "category"]).values)# 3 instrucciones 
columnas_transformadas=encoder.transform(titanic.select_dtypes(include=["object", "category"]).values).toarray()#


# In[ ]:


from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
columnas_transformadas_ordinales = encoder.fit_transform(titanic.select_dtypes(include=["object", "category"]).values)
columnas_transformadas_ordinales


# In[102]:


titanic[titanic.select_dtypes("object").columns]


# In[104]:


titanic[titanic.select_dtypes(include=["object", "category"]).columns]=columnas_transformadas_ordinales


# In[105]:


encoder.categories_


# In[106]:


#La importancia de apilar instrucciones 
# importancia de entender el apilamiento de instrucciones 


# ### Veamos como tengo el dataframe
# Nota : todos ya no tienen nulos y ya son flotantes 

# In[107]:


titanic.info()


# # Aprendamos a Separar variables 
# * Variable(s) predictoras
# * Variable(s) a predecir 

# In[ ]:


#Vamos a obtener las posibles variables para predecir un resultado
x=titanic.drop(columns="alive")
#separo la(s) columnas para predecir 
y=titanic["alive"]


# In[ ]:


#Como se ve X 
x.head(2)


# In[113]:


#Como se ve Y
y[:2]


# # Separar predictoras a predecir 
# Nota :solo se escala el valor de las variables predictoras

# In[125]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# mover librerias hacia arriba 


# In[130]:


df


# In[ ]:


titanic
#ya tiene selecion , analisis de completitu , analisis de objetois , imputado , onehotencoder, (dummy),Separe las variables 


# In[ ]:


#Solo para las variables categoricas 
#variable objetivo ( dicotomica (si o no , 0 y 1))
[ "class"]
#las variables que eran categoricas (strings)
["sex","embarked","who","embark_town"]
#Cuales eran las variables numericas continuas(1.2,8.3,etc)
["age","fare"]
# el resto ya eran numero entero(como tipo categoricas) (0,1,2,3)
["pclass","sibsp","parch","adult_male","alone"]


# In[ ]:


# Esta imputado 
import matplotlib.pyplot as plt

plt.hist(titanic["age"])
plt.show()


# In[135]:


#defino el metodo
sc=StandardScaler()
#Entreno para escalar
sc=sc.fit(titanic[["age","fare"]])
x_sc_scaled=sc.transform(titanic[["age","fare"]])
x_sc_scaled


# In[ ]:


plt.hist(x_sc_scaled[:,0])
plt.show()


# In[139]:


#defino el metodo
sc=MinMaxScaler()
#Entreno para escalar
sc=sc.fit(titanic[["age","fare"]])
x_sc_min_max=sc.transform(titanic[["age","fare"]])
x_sc_min_max


# In[140]:


plt.hist(x_sc_min_max[:,0])
plt.show()


# In[141]:


#defino el metodo
sc=StandardScaler()
#Entreno para escalar
sc=sc.fit(x[["age","fare"]])
x_sc_scaled=sc.transform(x[["age","fare"]])
x_sc_scaled


# In[142]:


x_sc_scaled.shape


# In[143]:


#como meto este escalado a el df 
x[["age","fare"]]=x_sc_scaled


# In[146]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,#Variables a las cuales ya les aplique el preprocesado
                                                     y, #la columna que tiene las repuesta (Si sobrevivio)
                                                     test_size=0.20,# Esto es de evaluacion (y es el 20% , entreno con el 80%) 
                                                     random_state=1996)#Para reproucibilidad del experiemnto ( Para que nos de los mismo resultados  )


# # Modelo de IA 

# 

# 

# 

# # Ejercicio semana 1
# *** Cree un analisis de completitud 
# 
# Cree una grafica de histograma con matplotliob de Titanic (variable numerica)
# Cree una grafica de barras con pandas del f titanic (variable categorica)
# Bonis 
# 
# Data: El dataset lo pueden obtener de seabon
# Criterios de aceptacion :
# Graficas con titulo
# Graficas con limites en eje x y y 
# Etiquetas (labels) en X y Y
# Bonus : Modificación de color y tamaño de letra
# 
# 

# #Ejercicio propuesto semana 2
# 
# 
# Dado el dataframe :
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# 
# haga todo el preprocesamiento 

# hola :
# Aunado a ello el proyecto 1 : con el cual se va a extender un reconocimiento de dominio de Analisis de datos con python :
# 
# # Ejercicio semana 1
# *** Cree un analisis de completitud 
# 
# Cree una grafica de histograma con matplotliob de Titanic (variable numerica)
# Cree una grafica de barras con pandas del f titanic (variable categorica)
# 
# 
# Data: El dataset lo pueden obtener de seabon
# Criterios de aceptacion :
# Graficas con titulo
# Graficas con limites en eje x y y 
# Etiquetas (labels) en X y Y
# Bonus : Modificación de color y tamaño de letra
# 
# 

# 
