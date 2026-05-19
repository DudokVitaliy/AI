# алгоритм кластеризації
from sklearn.cluster import KMeans
# створює тестові дані
from sklearn.datasets import make_blobs
# стандартизація ознак
from sklearn.preprocessing import StandardScaler
# побудова діаграм
import matplotlib.pyplot as plt
# робота з масивами
import numpy as np

# створення синтетичних даних
# 300 рандомних точок, 4 групи кластерів,
# 2 координат, 0.8 - розброс точок
# 42 - генерація даних однакових
X, y_true = make_blobs(n_samples=300,
                       centers=4,
                       n_features=2,
                       cluster_std=0.8,
                       random_state=42)

# Стандартизація
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# K-Means кластеризація
kmeans = KMeans(n_clusters=4,
                random_state=42,
                n_init=10)
labels = kmeans.fit_predict(x_scaled)

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# print(x_scaled)
# print(x_scaled[:,0])

# До кластеризації - це осі графіків
axes[0].scatter(x_scaled[:, 0], x_scaled[:, 1],
                alpha=0.6, c="gray")
axes[0].set_title("Оригінальні дані")
axes[0].grid(True)

# cmap - палітра кольорів
scatter = axes[1].scatter(x_scaled[:, 0], x_scaled[:, 1],
                          c=labels, cmap='viridis', s=50)
centers = scaler.transform(kmeans.cluster_centers_)
axes[1].scatter(centers[:, 0], centers[:, 1], c='red', s=200,
                marker='X', edgecolors='black', linewidths=2, label='Центроїди')
axes[1].set_title('K-Means кластеризація (k=4)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

