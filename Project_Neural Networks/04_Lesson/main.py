import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("Привіт! Робота з нейроною мережею")

# Завантаження дата set
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

# Нормалізація даних від 0 до 1
x_train = x_train/255.0
x_test = x_test/255.0

# Побудова моделі
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='relu'),
])
# Компіляція моделі
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# навчання
model.fit(x_train,y_train,epochs=15)

# Оцінка точності
test_loss, test_acc = model.evaluate(x_test,y_test)
print("Точність:", test_acc)

#Робимо прогноз
predicate = model.predict(x_test)

# Показуємо приклад
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Прогноз: {predicate[0].argmax()}")
plt.show()