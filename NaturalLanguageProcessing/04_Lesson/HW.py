import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN

#N1
X, y = make_blobs(
    n_samples=600,
    centers=5,
    cluster_std=1.2,
    random_state=0
)

#DataFrame
df = pd.DataFrame(X, columns=['income', 'spending_score'])

#Scatter
plt.figure(figsize=(6, 5))
plt.scatter(df['income'], df['spending_score'])
plt.title('Initial Data')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

#N2
kmeans = KMeans(n_clusters=5, random_state=0)
df['segment'] = kmeans.fit_predict(df[['income', 'spending_score']])

#Візуалізація K-Means
plt.figure(figsize=(6, 5))
plt.scatter(df['income'], df['spending_score'], c=df['segment'])
plt.title('K-Means Segmentation')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

#N3
dbscan = DBSCAN(eps=1.5, min_samples=10)
df['cluster_dbscan'] = dbscan.fit_predict(df[['income', 'spending_score']])

#Візуалізація DBSCAN
plt.figure(figsize=(6, 5))
plt.scatter(df['income'], df['spending_score'], c=df['cluster_dbscan'])
plt.title('DBSCAN Clustering')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

#N4
plt.figure(figsize=(12, 5))

#K-Means
plt.subplot(1, 2, 1)
plt.scatter(df['income'], df['spending_score'], c=df['segment'])
plt.title('K-Means Segmentation')
plt.xlabel('Income')
plt.ylabel('Spending Score')

#DBSCAN
plt.subplot(1, 2, 2)
plt.scatter(df['income'], df['spending_score'], c=df['cluster_dbscan'])
plt.title('DBSCAN Segmentation')
plt.xlabel('Income')
plt.ylabel('Spending Score')

#Збереження
plt.tight_layout()
plt.savefig('client_segmentation_comparison.png')

plt.show()