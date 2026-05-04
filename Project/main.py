# Ознаки і мітки
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import tree

# ----Завантаження даних-----
# x - ознаки, характеристики квітів
# y - правильні відповіді, тобто класи квітів
x, y = load_iris(return_X_y=True)
print("x:", x.shape)
print("y:", y.shape)

# Розподілення train/test
# 80% - даних для начання
# 20% - для тесту
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# створюємо 100 дерев для прийнятя рішень
#
fr = RandomForestClassifier(n_estimators=100)

# Навчання моделі
fr.fit(x_train, y_train)

# Візуалізація першого дерева з лісу
plt.figure(figsize=(20, 10))
tree.plot_tree(fr.estimators_[0],
               # feature_names="feature_names",
               # class_names="class_names",
               filled=True)
plt.show()
# Оцінка точності
accuracy = fr.score(x_test, y_test)

print("Accuracy:", accuracy)

