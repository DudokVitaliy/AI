import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, TensorDataset

data = load_breast_cancer()  # Дані для бінарної класифікації
X = data.data  # ознаки
y = data.target  # мітки (0 або 1)

# print(X.shape, y.shape)
# print("x = ", X)

# Проводимо нормалізацію даних.
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Маштабуємо дані

# print(X)

# Розділяємо дані на tain/test
# Тренеруємо модель
X_train, X_test, y_train, y_test = (
    train_test_split(X, y,
                     # 20% тестовий і 80% для навчання
                     test_size=0.2,
                     # при запуску, щоб був однакови результат
                     random_state=42))

# Перетворюємо дані у tensor
x_train = torch.tensor(X_train)
x_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Робимо DataLoader
train_dataset = TensorDataset(x_train, y_train)

train_loader = DataLoader(train_dataset,
                          batch_size=64,  # Розмір батчу
                          shuffle=True)  # Режим перемішування


# Робимо опис моделі
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        # Sequential — просто стек шарів
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),  # вхід → 64 нейрони
            nn.ReLU(),  # нелінійність

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)  # вихід: 1 значення (ймовірність)
            # ⚠️ Sigmoid НЕ додаємо тут (дивись нижче)
        )

    def forward(self, x):
        return self.model(x)


# --- 8. Ініціалізація ---
model = NeuralNet(X_train.shape[1])

# BCEWithLogitsLoss = Sigmoid + BCELoss (більш стабільно!)
criterion = nn.BCEWithLogitsLoss()

# Adam — один із найкращих оптимізаторів "з коробки"
optimizer = optim.Adam(model.parameters(), lr=0.001)
# --- 9. Навчання --
epochs = 50

for epoch in range(epochs):
    model.train()  # режим навчання

    total_loss = 0

    # Проходимо по батчах
    for batch_X, batch_y in train_loader:
        # Forward pass (прогноз)
        outputs = model(batch_X)

        # Обчислення loss
        loss = criterion(outputs, batch_y)

        # Обнулення градієнтів
        optimizer.zero_grad()

        # Backpropagation (обчислення градієнтів)
        loss.backward()

        # Оновлення ваг
        optimizer.step()

        total_loss += loss.item()

    # Вивід кожні 10 епох
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# --- 10. Оцінка моделі ---
model.eval()  # режим оцінки

with torch.no_grad():  # вимикаємо градієнти (швидше + менше пам’яті)

    logits = model(X_test)

    # Перетворення в ймовірності
    probs = torch.sigmoid(logits)

    # threshold = 0.5
    predicted = (probs > 0.5).float()

    accuracy = (predicted == y_test).sum().item() / y_test.size(0)

print(f"Accuracy: {accuracy:.4f}")

# --- 11. Збереження моделі ---
torch.save(model.state_dict(), "model.pth")

# --- 12. Завантаження моделі (якщо потрібно) ---
# new_model = NeuralNet(X_train.shape[1])
# new_model.load_state_dict(torch.load("model.pth"))
# new_model.eval()
