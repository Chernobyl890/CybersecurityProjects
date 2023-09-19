import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform

# Generate synthetic data with different sizes and densities
X1, y1 = make_blobs(n_samples=100, centers=[[0,0]], cluster_std=0.5, random_state=1)
X2, y2 = make_blobs(n_samples=500, centers=[[4,4]], cluster_std=1, random_state=1)
X3, y3 = make_blobs(n_samples=50, centers=[[-5,5]], cluster_std=2, random_state=1)
X = np.concatenate((X1, X2, X3), axis=0)

# Compute pairwise distances
D = squareform(pdist(X))

# Perform clustering with the Jarvis-Patrick algorithm
threshold = 2.5  # distance threshold for neighborhood
clusters = []
for i in range(X.shape[0]):
    neighborhood = np.where(D[i] < threshold)[0]
    if len(neighborhood) > 1:  # at least one neighbor is required to form a cluster
        clusters.append(neighborhood.tolist())

# Plot the clusters
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
plt.figure(figsize=(10, 8))
for i, cluster in enumerate(clusters):
    plt.scatter(X[cluster, 0], X[cluster, 1], c=colors[i % len(colors)], label='Cluster {}'.format(i+1))
plt.title('Jarvis-Patrick Clustering')
plt.legend()
plt.show()
