import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

print("="*20, " N1 ", "="*20)

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"].values

X = pd.get_dummies(X, drop_first=True)

print("Shape after encoding:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

x_train = torch.tensor(X_train, dtype=torch.float32)
x_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(
    TensorDataset(x_train, y_train),
    batch_size=32,
    shuffle=True
)

print("="*20, " N2 ", "="*20)

class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = MLP(X_train.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 30

losses = []
accuracies = []

print("="*20, " N3 ", "="*20)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        preds = model(batch_X)
        loss = criterion(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)

    model.eval()
    with torch.no_grad():
        test_preds = model(x_test)
        predicted = (test_preds > 0.5).float()
        acc = (predicted == y_test).float().mean().item()

    losses.append(epoch_loss)
    accuracies.append(acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {acc:.4f}")

metrics_df = pd.DataFrame({
    "Epoch": range(1, epochs + 1),
    "Loss": losses,
    "Accuracy": accuracies
})

print("\nFinal metrics:")
print(metrics_df.tail())

print("\n--- FINAL RESULT ---")
print(f"Loss: {losses[-1]:.4f}")
print(f"Accuracy: {accuracies[-1]:.4f}")

print("="*20, " N4 ", "="*20)

plt.figure()
plt.plot(losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.legend()
plt.savefig("loss_healthrisk_mlp.png")
plt.close()

plt.figure()
plt.plot(accuracies, label="Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over epochs")
plt.legend()
plt.savefig("accuracy_healthrisk_mlp.png")
plt.close()

print("Plots saved successfully!")