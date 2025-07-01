# Tarda bastante y es un proceso pesado
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import random 

# 1. Seleccionar categorías
categories = ['sci.space', 'rec.sport.baseball', 'talk.politics.mideast']

# 2. Cargar datos
train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

# 3. Crear pipeline: TF-IDF + Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB()) # aprendamos a hacer pipelines 

# 4. Entrenar modelo
model.fit(train_data.data, train_data.target)# Entrenamos 

# 5. Evaluar modelo
predicted = model.predict(test_data.data)# predecimos 
print(classification_report(test_data.target, predicted, target_names=categories))

# 6. Mostrar ejemplos de predicción
print("\n📄 Ejemplos de predicción:")
for i in random.sample(range(len(test_data.data)), 5):
    print(f"\nTexto:\n{test_data.data[i][:300]}...")
    print(f"Categoría real: {categories[test_data.target[i]]}")
    print(f"Predicción:     {categories[predicted[i]]}")