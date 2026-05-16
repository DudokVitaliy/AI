# PCA - метод головних компонентів
# будемо проводити зменшення розмірності даних із мінімальною втратою інформації

from sklearn.decomposition import PCA
# для стандартних даних
from sklearn.preprocessing import StandardScaler
# імпортуємо набір даних
from sklearn.datasets import load_iris
# бібліотека для графіки
import matplotlib.pyplot as plt

# завантаження даних
iris = load_iris()

# беремо ознаки
X = iris.data
y = iris.target

# стандартизація даних
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x_scaled)

print(f"Пояснена варіантність: {pca.explained_variance_ratio_}")
print(f"Сумарна варіантність: {pca.explained_variance_ratio_.sum():.2%}")

# Візуалізація
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('Iris Dataset - PCA проекція')
plt.colorbar(scatter)
plt.grid(True)
plt.show()

# Аналіз компонент
print("\nЗавантаження компонент:")
for i, component in enumerate(pca.components_):
    print(f"PC{i+1}: {component}")