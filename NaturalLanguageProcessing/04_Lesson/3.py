from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import time

# Завантаження даних
digits = load_digits()
X, y = digits.data, digits.target

print(f"Оригінальна розмірність: {X.shape}")

# Розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Стандартизація
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== КЛАСИФІКАЦІЯ БЕЗ ЗНИЖЕННЯ РОЗМІРНОСТІ ==========
print("\n1. БЕЗ зниження розмірності:")
start = time.time()
rf_full = RandomForestClassifier(n_estimators=100, random_state=42)
rf_full.fit(X_train_scaled, y_train)
y_pred_full = rf_full.predict(X_test_scaled)
time_full = time.time() - start

print(f"Точність: {accuracy_score(y_test, y_pred_full):.4f}")
print(f"Час навчання: {time_full:.3f} сек")

# ========== КЛАСИФІКАЦІЯ З ЗНИЖЕННЯМ РОЗМІРНОСТІ ==========
print("\n2. З зниженням розмірності (PCA на 20 компонент):")
pca = PCA(n_components=20)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Нова розмірність: {X_train_pca.shape}")
print(f"Пояснена варіантність: {pca.explained_variance_ratio_.sum():.2%}")

start = time.time()
rf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
rf_pca.fit(X_train_pca, y_train)
y_pred_pca = rf_pca.predict(X_test_pca)
time_pca = time.time() - start

print(f"Точність: {accuracy_score(y_test, y_pred_pca):.4f}")
print(f"Час навчання: {time_pca:.3f} сек")

# Порівняння
print(f"\n📊 Результати:")
print(f"Прискорення: {time_full/time_pca:.1f}x швидше")
print(f"Втрата точності: {(accuracy_score(y_test, y_pred_full) - accuracy_score(y_test, y_pred_pca))*100:.2f}%")