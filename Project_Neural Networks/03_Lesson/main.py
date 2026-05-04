import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# -------Завантаження і підготовка DATASET------
(X_train, y_train), (X_test, y_test) =  mnist.load_data()
print(f"Форма для тренування датасет: {X_train.shape}")
print(f"Форма тестового датасету: {X_test.shape}")
print(f"Діапазон значень пікселів: [{X_train.min()}, {X_train.max()}]")
print(f"Унікальні класи: {np.unique(y_train)}")

# Нормалізація пікселів [0, 255] → [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"\nПісля нормалізації - діапазон: [{X_train.min()}, {X_train.max()}]")

# Розширити розміри для конволюційної сітки (додати канал)
# З (60000, 28, 28) в (60000, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Форма після додання каналу: {X_train.shape}")

# Розділити тренувальний датасет на тренування та валідацію (80/20)
split_idx = int(0.8 * len(X_train))
X_train_split = X_train[:split_idx]
X_val = X_train[split_idx:]
y_train_split = y_train[:split_idx]
y_val = y_train[split_idx:]

print(f"\nРозділення датасету:")
print(f"  Тренування: {X_train_split.shape[0]} прикладів")
print(f"  Валідація: {X_val.shape[0]} прикладів")
print(f"  Тест: {X_test.shape[0]} прикладів")

# Будуємо архітектурні модель
print("\n" + "=" * 70)
print("2. ПОБУДОВА АРХІТЕКТУРИ CNN")
print("=" * 70)

model = models.Sequential([
    # Перший згортковий блок
    layers.Conv2D(
        filters=32,  # Кількість фільтрів (ознак)
        kernel_size=(3, 3),  # Розмір фільтру (3×3)
        padding='same',  # Додавання паддінгу (зберігає розмір)
        activation='relu',  # ReLU активація
        input_shape=(28, 28, 1)  # Форма входу
    ),
    layers.BatchNormalization(),  # Батч-нормалізація
    layers.MaxPooling2D(pool_size=(2, 2)),  # Максимальний пулінг 2×2

    # Другий згортковий блок
    layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Третій згортковий блок (опціонально для більшої складності)
    layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten для переходу до повносвязних шарів
    layers.Flatten(),

    # Повносвязні шари (Dense)
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Dropout для регуляризації (50% нейронів)

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),  # Dropout 30%

    # Вихідний шар для мультикласової класифікації
    layers.Dense(10, activation='softmax')  # 10 класів (0-9)
])

# Вивести архітектуру мережі
print("\nАрхітектура моделі:")
model.summary()

# Компілюємо модель
print("\n" + "=" * 70)
print("3. КОМПІЛЯЦІЯ МОДЕЛІ")
print("=" * 70)

optimizer = Adam(learning_rate=0.001)  # Адаптивний оптимізатор

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',  # Функція втрат для мультикласової класифікації
    metrics=['accuracy']  # Метрика для відстеження
)

print("✓ Модель скомпільована успішно")
print(f"  Оптимізатор: Adam (learning_rate=0.001)")
print(f"  Функція втрат: Sparse Categorical Cross-Entropy")
print(f"  Метрики: Accuracy")

# проводимо навчання моделі

print("\n" + "=" * 70)
print("4. НАВЧАННЯ МОДЕЛІ З БАТЧИНГОМ")
print("=" * 70)

# Параметри батчингу
batch_size = 128  # Розмір батчу
epochs = 20  # Кількість епох

print(f"\nПараметри батчингу:")
print(f"  Розмір батчу: {batch_size}")
print(f"  Всього батчів на епоху: {len(X_train_split) // batch_size}")
print(f"  Кількість епох: {epochs}")
print(f"  Всього ітерацій: {epochs * (len(X_train_split) // batch_size)}")

# Ранню зупинку для запобігання перенастроюванню
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Навчання моделі
history = model.fit(
    X_train_split, y_train_split,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# Оцінка моделі

print("\n" + "=" * 70)
print("5. ОЦІНКА МОДЕЛІ НА ТЕСТОВОМУ НАБОРІ")
print("=" * 70)

# Оцінка на тестовому наборі
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nРезультати на тестовому наборі:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_accuracy * 100:.2f}%")

# Передбачення для всього тестового набору
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Деталізована оцінка
print("\n" + "-" * 70)
print("Детальна класифікаційна звіт:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

#Аналіз помилок і матриця помилок
print("\n" + "=" * 70)
print("6. АНАЛІЗ ПОМИЛОК")
print("=" * 70)

# Матриця помилок
cm = confusion_matrix(y_test, y_pred)

# Знайти помилки
errors = y_test != y_pred
error_indices = np.where(errors)[0]
num_errors = len(error_indices)

print(f"\nВсього помилок: {num_errors} з {len(y_test)} ({num_errors / len(y_test) * 100:.2f}%)")
print(f"Правильних передбачень: {len(y_test) - num_errors} ({(1 - num_errors / len(y_test)) * 100:.2f}%)")

# Найпроблемніші цифри
print("\nПроблемні цифри (найбільш часто помиляються):")
for i in range(10):
    wrong_as_i = np.sum((y_test == i) & (y_pred != i))
    total_i = np.sum(y_test == i)
    if total_i > 0:
        error_rate = wrong_as_i / total_i * 100
        print(f"  Цифра {i}: {wrong_as_i}/{total_i} помилок ({error_rate:.1f}%)")

#Візуалізація результатів
print("\n" + "=" * 70)
print("7. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Графік Loss під час навчання
ax = axes[0, 0]
ax.plot(history.history['loss'], label='Training Loss', marker='o')
ax.plot(history.history['val_loss'], label='Validation Loss', marker='s')
ax.set_xlabel('Епоха')
ax.set_ylabel('Loss')
ax.set_title('Динаміка Loss під час навчання')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Графік Accuracy під час навчання
ax = axes[0, 1]
ax.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
ax.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
ax.set_xlabel('Епоха')
ax.set_ylabel('Accuracy')
ax.set_title('Динаміка Accuracy під час навчання')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Матриця помилок
ax = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_xlabel('Передбачена цифра')
ax.set_ylabel('Справжня цифра')
ax.set_title('Матриця помилок (Confusion Matrix)')

# 4. Розподіл довіри моделі
ax = axes[1, 1]
correct_mask = y_test == y_pred
correct_probs = y_pred_probs[correct_mask].max(axis=1)
incorrect_probs = y_pred_probs[~correct_mask].max(axis=1)

ax.hist(correct_probs, bins=30, alpha=0.7, label='Правильні')
ax.hist(incorrect_probs, bins=30, alpha=0.7, label='Неправильні')
ax.set_xlabel('Максимальна ймовірність (довіра)')
ax.set_ylabel('Кількість прикладів')
ax.set_title('Розподіл довіри моделі')
ax.legend()

plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("✓ Графіки збережені в 'training_results.png'")

# Передбачення
print("\n" + "=" * 70)
print("8. ПРИКЛАДИ ПЕРЕДБАЧЕНЬ")
print("=" * 70)

# Показати деякі правильні та неправильні передбачення
fig, axes = plt.subplots(3, 5, figsize=(15, 9))

# Неправильні передбачення
print("\nПримери НЕПРАВИЛЬНИХ передбачень:")
ax_idx = 0
for idx in error_indices[:5]:
    ax = axes[0, ax_idx]
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f"Справжня: {y_test[idx]}\nПередбач: {y_pred[idx]}",
                 color='red', fontweight='bold')
    ax.axis('off')
    print(f"  Індекс {idx}: Справжня цифра {y_test[idx]}, модель передбачила {y_pred[idx]}")
    ax_idx += 1

# Правильні передбачення з низькою довірою
print("\nПримери ПРАВИЛЬНИХ передбачень з НИЗЬКОЮ довірою:")
correct_indices = np.where(y_test == y_pred)[0]
low_confidence_mask = np.argsort(y_pred_probs[correct_indices].max(axis=1))[:5]
low_conf_indices = correct_indices[low_confidence_mask]

ax_idx = 0
for idx in low_conf_indices:
    ax = axes[1, ax_idx]
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    confidence = y_pred_probs[idx].max() * 100
    ax.set_title(f"Цифра: {y_test[idx]}\nДовіра: {confidence:.1f}%",
                 color='orange', fontweight='bold')
    ax.axis('off')
    print(f"  Індекс {idx}: Цифра {y_test[idx]}, довіра {confidence:.1f}%")
    ax_idx += 1

# Правильні передбачення з високою довірою
print("\nПримери ПРАВИЛЬНИХ передбачень з ВИСОКОЮ довірою:")
high_confidence_mask = np.argsort(y_pred_probs[correct_indices].max(axis=1))[-5:]
high_conf_indices = correct_indices[high_confidence_mask]

ax_idx = 0
for idx in high_conf_indices:
    ax = axes[2, ax_idx]
    ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    confidence = y_pred_probs[idx].max() * 100
    ax.set_title(f"Цифра: {y_test[idx]}\nДовіра: {confidence:.1f}%",
                 color='green', fontweight='bold')
    ax.axis('off')
    print(f"  Індекс {idx}: Цифра {y_test[idx]}, довіра {confidence:.1f}%")
    ax_idx += 1

plt.suptitle('Приклади передбачень моделі', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('predictions_examples.png', dpi=150, bbox_inches='tight')
print("\n✓ Приклади збережені в 'predictions_examples.png'")


print("\n" + "=" * 70)
print("9. ВІЗУАЛІЗАЦІЯ CONVOLUTIONAL ФІЛЬТРІВ")
print("=" * 70)

# Отримати ваги першого конволюційного шару
first_conv_layer_weights = model.layers[0].get_weights()[0]  # (3, 3, 1, 32)

print(f"Форма ваг першого Conv-шару: {first_conv_layer_weights.shape}")
print(f"  (kernel_height, kernel_width, input_channels, num_filters)")

# Візуалізувати перші 16 фільтрів
fig, axes = plt.subplots(4, 8, figsize=(14, 7))
axes = axes.flatten()

for i in range(32):  # 32 фільтри в першому шарі
    filter_weights = first_conv_layer_weights[:, :, 0, i]

    # Нормалізувати для кращої візуалізації
    filter_min = filter_weights.min()
    filter_max = filter_weights.max()
    filter_normalized = (filter_weights - filter_min) / (filter_max - filter_min + 1e-8)

    axes[i].imshow(filter_normalized, cmap='viridis')
    axes[i].set_title(f'Фільтр {i}', fontsize=8)
    axes[i].axis('off')

plt.suptitle('Вивчені конволюційні фільтри першого шару (32×3×3)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/conv_filters.png', dpi=150, bbox_inches='tight')
print("✓ Фільтри збережені в 'conv_filters.png'")

# 10. ФІНАЛЬНИЙ ЗВІТ
#
print("\n" + "=" * 70)
print("ФІНАЛЬНИЙ ЗВІТ")
print("=" * 70)

print(f"""
📊 Результати навчання CNN для MNIST:

   Архітектура:
   - Вхід: 28×28×1 (чорно-біле зображення)
   - Conv1: 32 фільтри 3×3 + BatchNorm + MaxPool
   - Conv2: 64 фільтри 3×3 + BatchNorm + MaxPool
   - Conv3: 128 фільтри 3×3 + BatchNorm + MaxPool
   - Dense1: 256 нейронів + BatchNorm + Dropout(0.5)
   - Dense2: 128 нейронів + BatchNorm + Dropout(0.3)
   - Вихід: 10 нейронів (Softmax)

   Загальна кількість параметрів: {model.count_params():,}

   Результати:
   - Точність на тесті: {test_accuracy * 100:.2f}%
   - Loss на тесті: {test_loss:.4f}
   - Помилок: {num_errors}/{len(y_test)}

   Процес навчання:
   - Батч-розмір: {batch_size}
   - Епох виконано: {len(history.history['loss'])}
   - Оптимізатор: Adam (lr=0.001)
   - Ранню зупинку: активована (patience=5)

   📈 Графіки та図:
   - training_results.png — динаміка loss/accuracy
   - predictions_examples.png — приклади передбачень
   - conv_filters.png — вивчені конволюційні фільтри
""")

print("=" * 70)
print("✓ Програма успішно завершена!")
print("=" * 70)