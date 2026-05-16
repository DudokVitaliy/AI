# Робота з SVD - зменшення розмірності даних та їх візуалізація
# Будемо проводити сингулярне розпадання даних

import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

np.random.seed(42) #генерація даних буде однакова при кожному запуску
# 100 даних із 50 ознаками (матриця)
data = np.random.randn(100, 50)

# розклад даних - із 50 ознак вибираємо дві головних
svd = TruncatedSVD(n_components=2)
# зміна розмірності було 100 на 50 - стало 100 на 2
reduced_data = svd.fit_transform(data)
# вивід варіантів
print(f"Пояснення варіантності: {svd.explained_variance_ratio_.sum():.2%}")
#у нас 11.07% - інші дані втрачені
# чим вище значення - тим важливіший напрямок у даних
print(f"Сингулярні значення: {svd.singular_values_[:5]}")
# створюємо графік розміром 10х6
plt.figure(figsize=(10, 6))
# малюємо точки по x та y
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.6)
# додаємо мітки
plt.xlabel('SVD компонента 1')
plt.ylabel('SVD компонента 2')
plt.title('Дані після SVD розкладення')
plt.grid(True)
plt.show()

