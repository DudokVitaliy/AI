import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("="*20, " N1", "="*20)
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
r2_dt = r2_score(y_test, y_pred_dt)

importances = pd.Series(dt.feature_importances_, index=X.columns)

plt.figure(figsize=(8,5))
importances.sort_values().plot(kind="barh")
plt.title("Decision Tree Feature Importances")
plt.show()

print("="*20, " N2", "="*20)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
r2_rf = r2_score(y_test, y_pred_rf)


print("="*20, " N3", "="*20)
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
r2_xgb = r2_score(y_test, y_pred_xgb)


results_reg = pd.DataFrame({
    "Model": ["DecisionTreeRegressor", "RandomForestRegressor", "XGBRegressor"],
    "Test R²": [r2_dt, r2_rf, r2_xgb]
})

print("\n=== REGRESSION RESULTS ===")
print(results_reg)


print("="*20, " N4", "="*20)
cancer = load_breast_cancer()
Xc = pd.DataFrame(cancer.data, columns=cancer.feature_names)
yc = cancer.target

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42
)

log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(Xc_train, yc_train)

yc_pred = log_reg.predict(Xc_test)

accuracy = accuracy_score(yc_test, yc_pred)
precision = precision_score(yc_test, yc_pred)
recall = recall_score(yc_test, yc_pred)
f1 = f1_score(yc_test, yc_pred)

print("\n=== CLASSIFICATION METRICS ===")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

cm = confusion_matrix(yc_test, yc_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()