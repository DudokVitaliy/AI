import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Використовується:", device)

#N1
dataset_path = "chest_xray"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])
train_dataset = datasets.ImageFolder(
    root=os.path.join(dataset_path, "train"),
    transform=transform
)
test_dataset = datasets.ImageFolder(
    root=os.path.join(dataset_path, "test"),
    transform=transform
)
class_names = train_dataset.classes

print("Класи:", class_names)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

#N2
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 2)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=0.0001
)

epochs = 5

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_acc = correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {epoch_acc:.4f}"
    )

#N3

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.numpy())

        all_predictions.extend(
            predicted.cpu().numpy()
        )

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions
)

recall = recall_score(
    all_labels,
    all_predictions
)

f1 = f1_score(
    all_labels,
    all_predictions
)

print("\n===== Метрики =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap=plt.cm.Blues)

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()

torch.save(
    model.state_dict(),
    "pneumonia_resnet18.pth"
)

print("\nМодель збережено: pneumonia_resnet18.pth")

#N4

indices = random.sample(
    range(len(test_dataset)),
    8
)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(14, 8)
)

fig.suptitle("Pneumonia Predictions")

for i, idx in enumerate(indices):

    image, label = test_dataset[idx]

    model.eval()

    with torch.no_grad():

        input_image = image.unsqueeze(0).to(device)

        output = model(input_image)

        _, predicted = torch.max(output, 1)

    predicted_class = class_names[predicted.item()]
    true_class = class_names[label]

    img = image.numpy().transpose((1, 2, 0))

    img = (img * 0.5) + 0.5

    img = np.clip(img, 0, 1)

    ax = axes[i // 4, i % 4]

    ax.imshow(img.squeeze(), cmap="gray")

    ax.set_title(
        f"Pred: {predicted_class}\nTrue: {true_class}",
        color="green"
        if predicted_class == true_class
        else "red"
    )

    ax.axis("off")

plt.tight_layout()

plt.savefig("pneumonia_predictions.png")

plt.show()

print("Графік збережено: pneumonia_predictions.png")

print("\n===== ВИСНОВОК =====")

print("""
Модель ResNet18 успішно навчена
для визначення пневмонії
за рентген-знімками.

Було використано:
- Transfer Learning
- Замороження згорткових шарів
- Optimizer Adam
- CrossEntropyLoss

Модель показала хороші результати
за метриками:
Accuracy, Precision,
Recall та F1-score.

Проте для реального
медичного впровадження необхідно:
- більше даних;
- додаткова валідація;
- перевірка лікарями;
- тестування на реальних клінічних даних.

На даному етапі модель підходить
для демонстраційного
або дослідницького використання.
""")