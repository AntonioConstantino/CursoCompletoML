# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 21:37:23 2025

@author: jacj2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generar datos de muestra
X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.5, random_state=42)

#Preprocesamiento # pasos previos omitidos 

# Aplicar clustering jerárquico
Z = linkage(X, method='ward')

# Visualizar dendrograma
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Dendrograma de Clustering Jerárquico")
plt.show()