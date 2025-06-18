import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.datasets import get_rdataset

# Cargar el dataset AirPassengers
data = get_rdataset("AirPassengers").data
data['Month'] = pd.date_range(start='1949-01', periods=len(data), freq='M')
data.set_index('Month', inplace=True)

# Visualizar la serie temporal
data.plot(title='Pasajeros Aéreos Mensuales')
plt.ylabel('Miles de pasajeros')
plt.show()

# Ajustar un modelo ARIMA (p=2, d=1, q=2 como ejemplo)
model = ARIMA(data['value'], order=(2, 1, 2))
model_fit = model.fit()

# Mostrar resumen del modelo
print(model_fit.summary())

# Hacer predicciones
forecast = model_fit.forecast(steps=12)
forecast.plot(title='Pronóstico de pasajeros')
plt.show()