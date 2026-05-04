import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score


print("=== N1 ===")

#датасет
california = fetch_california_housing(as_frame=True)
df = california.frame

X = df[['MedInc']].values
y = df['MedHouseVal'].values

#поліноміальні ознаки ступеня 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

#розділення на train/test
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

#навчання моделі
model_poly = LinearRegression()
model_poly.fit(X_train, y_train)

#R2 на тесті
r2_test = r2_score(y_test, model_poly.predict(X_test))
print(f"R2 на тестовій вибірці (deg=2): {r2_test:.3f}")

print("\n=== N2 ===")
degrees = [1, 2, 3]
results = []

for deg in degrees:
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(df[['MedInc']])
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))

    results.append([deg, r2_train, r2_test])

df_poly_results = pd.DataFrame(results, columns=["Ступінь полінома", "R² Train", "R² Test"])
print(df_poly_results)

print("\n=== N3 ===")

features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']
X = df[features].values
y = df['MedHouseVal'].values
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

print(f"Linear Regression R2 на тесті: {r2_lin:.3f}")
print(f"Lasso R2 на тесті: {r2_lasso:.3f}")
print(f"Кількість коефіцієнтів, які Lasso занулила: {zero_coeff}")

print("\n=== N4 ===")

#Ridge
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
r2_ridge = r2_score(y_test, ridge.predict(X_test))
non_zero_ridge = np.sum(ridge.coef_ != 0)

#таблиця
comparison = pd.DataFrame({
    "Модель": ["LinearRegression", "Ridge(alpha=1.0)"],
    "R2 Test": [r2_lin, r2_ridge],
    "Кількість ненульових коеф": [len(lin_reg.coef_), non_zero_ridge]
})
print(comparison)