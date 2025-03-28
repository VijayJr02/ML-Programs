import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = sns.load_dataset("penguins").dropna()
X = StandardScaler().fit_transform(data[["bill_length_mm", "bill_depth_mm"]])

inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in range(1, 11)]
plt.plot(range(1, 11), inertia, 'o--')
plt.show()

kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
plt.scatter(*kmeans.cluster_centers_.T, c='red', marker='X', s=200)
plt.show()
