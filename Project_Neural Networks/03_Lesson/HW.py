import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd

#налаштування
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 10

#дані
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#модель без нормалізації
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

#модель з нормалізацією
class StableCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

#функції
def train_model(model, use_clip=False):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()

            # Gradient Clipping
            if use_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return losses


def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    loss_total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_total += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    loss_avg = loss_total / len(test_loader)

    return accuracy, loss_avg


#навчання
print("\n--- Навчання без нормалізації ---")
model1 = SimpleCNN()
losses1 = train_model(model1)
acc1, loss1 = evaluate(model1)

print("\n--- Навчання з BatchNorm + Dropout + Gradient Clipping ---")
model2 = StableCNN()
losses2 = train_model(model2, use_clip=True)
acc2, loss2 = evaluate(model2)

#графік
plt.figure(figsize=(8, 5))
plt.plot(losses1, label="Без нормалізації")
plt.plot(losses2, label="BatchNorm + Dropout")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Порівняння моделей")
plt.legend()
plt.grid()
plt.savefig("comparison.png")
plt.show()

#таблиця
data = {
    "Конфігурація": ["Без нормалізації", "BatchNorm + Dropout"],
    "Accuracy": [acc1, acc2],
    "Loss": [loss1, loss2],
    "Примітка": ["вихідна модель", "поліпшена стабільність"]
}

df = pd.DataFrame(data)

print("\nТАБЛИЦЯ ПОРІВНЯННЯ:\n")
print(df)

df.to_csv("comparison.csv", index=False)