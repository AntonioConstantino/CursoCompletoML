# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 21:34:48 2025

@author: jacj2
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generar datos de muestra
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=3, random_state=42)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=50)
plt.title("Clustering con K-Means")
plt.show()
# rama de prueba