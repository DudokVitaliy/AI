import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

print("="*20, " N1 ", "="*20)
#дані
X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)

#регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

#поліноміальні ознаки ступеня 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

#модель на поліноміальних ознаках
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

#візуалізація
plt.scatter(X, y, color='blue', label='Дані')
plt.plot(X, y_pred_lin, color='green', label='Лінійна регресія')
plt.plot(X, y_pred_poly, color='red', label='Поліноміальна регресія (deg=2)')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Лінійна vs Поліноміальна регресія (deg=2)")
plt.show()

print("="*20, " N2 ", "="*20)
#дані
X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degrees = [2, 5, 10]
results = []

for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    r2_train = r2_score(y_train, model.predict(X_train_poly))
    r2_test = r2_score(y_test, model.predict(X_test_poly))

    results.append([deg, r2_train, r2_test])

#таблиця
df = pd.DataFrame(results, columns=["Ступінь полінома", "R² Train", "R² Test"])
print(df)

print("="*20, " N3 ", "="*20)
#дані
X, y = make_regression(n_samples=500, n_features=20, n_informative=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
r2_lin = r2_score(y_test, lin_reg.predict(X_test))

#Lasso
lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)
r2_lasso = r2_score(y_test, lasso.predict(X_test))
zero_coeff = np.sum(lasso.coef_ == 0)

print(f"Linear Regression R² на тесті: {r2_lin:.3f}")
print(f"Lasso R² на тесті: {r2_lasso:.3f}")
print(f"Кількість коефіцієнтів, які Lasso занулила: {zero_coeff}")

print("="*20, " N4 ", "="*20)
#використовуємо ті ж дані, що і для Lasso
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
r2_ridge = r2_score(y_test, ridge.predict(X_test))
non_zero_ridge = np.sum(ridge.coef_ != 0)

#таблиця
comparison = pd.DataFrame({
    "Модель": ["LinearRegression", "Ridge(alpha=1.0)"],
    "R² Test": [r2_lin, r2_ridge],
    "Кількість ненульових коеф.": [len(lin_reg.coef_), non_zero_ridge]
})

print(comparison)