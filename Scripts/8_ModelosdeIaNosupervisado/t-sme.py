# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 21:57:29 2025

@author: jacj2
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits

# Cargar dataset de dígitos
digits = load_digits()
X = digits.data
y = digits.target

# Aplicar t-SNE para reducir a 2 dimensiones
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# Graficar los datos reducidos
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='jet', edgecolor='k')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.title('Dígitos en espacio t-SNE')
plt.colorbar(label='Dígito')
plt.show()