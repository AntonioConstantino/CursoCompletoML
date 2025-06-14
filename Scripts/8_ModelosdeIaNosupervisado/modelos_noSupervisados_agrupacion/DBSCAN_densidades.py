# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 21:33:53 2025

@author: jacj2
"""
#esto es un ejemplo de commit en github 
# este algoritmo necesita densidades ( que los puntos esten juntos)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN# densidades 
from sklearn.datasets import make_blobs # Hacer datos ficticios

# Generar datos de muestra
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1, random_state=42)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=1,#parametro distancia 
                min_samples=15)# Numero de puntitos juntos minimo para hacer un cluster 
labels = dbscan.fit_predict(X)

# Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50)
plt.title("Clustering con DBSCAN")
plt.show()
