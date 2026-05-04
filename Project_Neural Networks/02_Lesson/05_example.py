import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


model = NeuralNetwork()
print(model)
# Функція втрат для вимірювання похибки.
criterion = nn.BCELoss()
# Оптимізатор для оновлення ваг на основі обчислених градієнтів.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# optimizer.zero_grad() очищає накопичені градієнти з попереднього кроку.
# Функція Forward Pass(model(inputs)) передає вхідні дані через модель для генерації прогнозів.
# Обчислення втрат (критерій(виходи, цілі)) обчислює різницю між прогнозами та фактичними мітками.
# Зворотне поширення (loss.backward()) обчислює градієнти для всіх вагових коефіцієнтів.
# Крок оптимізатора (optimizer.step()) оновлює ваги на основі обчислених градієнтів.

inputs = torch.randn((100, 10))
targets = torch.randint(0, 2, (100, 1)).float()
epochs = 20

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Застосування
# Комп'ютерний зір : PyTorch широко використовується для класифікації зображень, виявлення об'єктів та сегментації за допомогою CNN та трансформаторів (наприклад, ViT).
# Обробка природної мови (NLP) : PyTorch підтримує трансформатори, рекурентні нейронні мережі (RNN) та LSTM для таких застосувань, як генерація тексту та аналіз настроїв.
# Навчання з підкріпленням : PyTorch використовується в глибоких Q-мережах (DQN), методах градієнта політики та алгоритмах актор-критик.