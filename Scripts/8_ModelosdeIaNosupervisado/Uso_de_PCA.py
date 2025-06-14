# -*- coding: utf-8 -*-
"""
Editor de Spyder

@Antonio constantino
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Cargar dataset de dígitos
digits = load_digits()
X = digits.data
y = digits.target

# Aplicar PCA para reducir a 2 dimensiones
pca = PCA(
    n_components=6# NUemro de columnas 
          )# metodo basado en varianza ( metodo lineal varianza)
X_pca = pca.fit_transform(X)#Entrena el modelo y tranforma para el numero de componentes 

# Graficar los datos reducidos
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.title('Dígitos en espacio PCA')
plt.colorbar(label='Dígito')
plt.show()

#Bosting (arboles)
#random forest importances (clasificacion )

# Entrenar modelo de IA 
#LDA=70%
#PCA=80%