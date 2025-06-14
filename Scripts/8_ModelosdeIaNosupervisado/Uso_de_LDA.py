# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 21:50:53 2025

@author: jacj2
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_digits

# Cargar dataset de dígitos
digits = load_digits()
X = digits.data
y = digits.target

# Aplicar LDA para reducir a 2 dimensiones
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Graficar los datos reducidos
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='plasma', edgecolor='k')
plt.xlabel('Componente discriminante 1')
plt.ylabel('Componente discriminante 2')
plt.title('Dígitos en espacio LDA')
plt.colorbar(label='Dígito')
plt.show()