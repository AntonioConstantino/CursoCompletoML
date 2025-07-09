from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar datos
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.3, random_state=42
)

from sklearn.model_selection import GridSearchCV

# Definir el modelo
rf = RandomForestClassifier(random_state=42)

# Definir la grilla de hiperparámetros
#Se ajusta en funcion del metodo y de los parametrso que permita el metodo , ver documentacion 
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=rf,# metodo de machine learning  
                           param_grid=param_grid,# cuadricula de busqueda 
                           cv=5, #validacion cruzada  
                           scoring='accuracy', # en clasificacion , metricas de clasif , en regresion , metodos de regresion
                           n_jobs=-1) # cuantos nucleos del procesaor necesito usar 

# Entrenar
grid_search.fit(X_train, y_train)

# Resultados
print("Mejores parámetros (GridSearch):", grid_search.best_params_)
print("Precisión en test:", accuracy_score(y_test, grid_search.best_estimator_.predict(X_test)))
"""
!pip install scikit-optimize

from skopt import BayesSearchCV
from skopt.space import Integer, Categorical

# Definir el espacio de búsqueda
search_space = {
    'n_estimators': Integer(50, 200),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(2, 10)
}

# Configurar BayesSearchCV
bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    search_spaces=search_space,
    n_iter=25,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Entrenar
bayes_search.fit(X_train, y_train)

# Resultados
print("Mejores parámetros (Bayesian Search):", bayes_search.best_params_)
print("Precisión en test:", accuracy_score(y_test, bayes_search.best_estimator_.predict(X_test)))"""