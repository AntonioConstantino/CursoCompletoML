# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 21:59:10 2025

@author: jacj2
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.datasets import load_digits

# Cargar dataset de dígitos
digits = load_digits()
X = digits.data
y = digits.target

# Aplicar Isomap para reducir a 2 dimensiones
isomap = Isomap(n_components=2, n_neighbors=5)
X_isomap = isomap.fit_transform(X)

# Graficar los datos reducidos
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y, cmap='coolwarm', edgecolor='k')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.title('Dígitos en espacio Isomap')
plt.colorbar(label='Dígito')
plt.show()