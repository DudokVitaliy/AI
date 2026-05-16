import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#N1

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#N2

pca_2 = PCA(n_components=2)
X_train_pca_2 = pca_2.fit_transform(X_train)

plt.figure()
for label in np.unique(y_train):
    plt.scatter(
        X_train_pca_2[y_train == label, 0],
        X_train_pca_2[y_train == label, 1],
        label=f"Class {label}"
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Components Visualization")
plt.legend()
plt.show()

#N3

components_list = [2, 5, 10]
accuracies = []

for n in components_list:
    pca = PCA(n_components=n)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)

    print(f"Components: {n}, Accuracy: {acc:.4f}")

#N4

plt.figure()
plt.plot(components_list, accuracies, marker='o')

plt.xlabel("Number of Components")
plt.ylabel("Accuracy")
plt.title("PCA Accuracy vs Components")

plt.savefig("pca_accuracy_results.png")
plt.show()