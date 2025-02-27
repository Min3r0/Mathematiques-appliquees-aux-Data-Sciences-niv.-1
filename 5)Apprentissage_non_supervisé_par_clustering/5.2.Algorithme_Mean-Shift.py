import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=6, cluster_std=2.0, random_state=42)

bandwidth = estimate_bandwidth(X, quantile=0.2)

meanshift = MeanShift(bandwidth=bandwidth)
meanshift.fit(X)

labels = meanshift.labels_
centers = meanshift.cluster_centers_

# Affichage des clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label="Centres")
plt.legend()
plt.title("Clustering Mean-Shift")
plt.show()
