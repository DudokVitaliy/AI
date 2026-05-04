import torch
import torch.nn as nn
import torch.optim as optim

#Дані

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

#Модель

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

model = SimpleNet()

#Функція втрат і оптимізатор

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

#Навчання

for epoch in range(1000):
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, y)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#Результат

print("\nРезультати після навчання:")
with torch.no_grad():
    predictions = model(X)
    print(predictions)