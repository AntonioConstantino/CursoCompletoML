#! pip install tensorflow keras-tuner scikit-learn
import numpy as np
from sklearn.datasets import load_digit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# 1. Cargar y preparar los datos
digits = load_digits()
X = digits.data
y = digits.target

# Escalar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 2. Definir el modelo con hiperparámetros
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))

    # Número de capas ocultas: entre 1 y 3
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
                activation="relu"
            )
        )

    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 3. Configurar el tuner ( bayesian y random )
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=10,
    executions_per_trial=1,
    directory="my_tuner_dir",
    project_name="digit_classification"
)

# 4. Ejecutar la búsqueda
tuner.search(X_train, y_train, epochs=20, validation_split=0.2, verbose=1)

# 5. Obtener el mejor modelo
best_model = tuner.get_best_models(num_models=1)[0]

# 6. Evaluar en el conjunto de prueba
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f"\n✅ Precisión en test: {test_acc:.4f}")

# 7. Mostrar los mejores hiperparámetros
best_hps = tuner.get_best_hyperparameters(1)[0]
print("\n🎯 Mejores hiperparámetros encontrados:")
print(f"Capas ocultas: {best_hps.get('num_layers')}")
for i in range(best_hps.get('num_layers')):
    print(f" - Neuronas en capa {i+1}: {best_hps.get(f'units_{i}')}")
print(f"Learning rate: {best_hps.get('learning_rate')}")