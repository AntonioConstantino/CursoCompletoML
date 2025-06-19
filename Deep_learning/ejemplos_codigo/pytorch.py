import torch
import torch.nn as nn
import torch.optim as optim

# Datos dummy
x_train = torch.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
y_train = torch.tensor([[1.0], [3.0], [5.0], [7.0], [9.0]])

# Modelo: una sola neurona (una capa lineal)
class OneNeuronModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # input dim = 1, output dim = 1

    def forward(self, x):
        return self.linear(x)

model = OneNeuronModel()

# Función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entrenamiento del modelo
for epoch in range(500):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()#ajuste de pesos ( buscar )
    optimizer.step()

# Prueba de predicción
x_test = torch.tensor([[10.0]])
prediccion = model(x_test).item()
print(f"Predicción para x = 10: {prediccion:.2f}")