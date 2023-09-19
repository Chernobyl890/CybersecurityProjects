import numpy as np
import matplotlib.pyplot as plt
import hdbscan

# Generate synthetic data with three clusters of different densities
X1 = np.random.normal(loc=[0, 0], scale=[0.3, 0.3], size=(100, 2))
X2 = np.random.normal(loc=[2, 2], scale=[0.1, 0.1], size=(50, 2))
X3 = np.random.normal(loc=[-2, 2], scale=[0.5, 0.5], size=(20, 2))
X = np.vstack([X1, X2, X3])

# Add some noise/outliers
noise = np.random.uniform(low=-4, high=4, size=(30, 2))
X = np.vstack([X, noise])

# Perform density-based clustering using DENCLUE
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
clusterer.fit(X)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=clusterer.labels_, cmap='viridis')
plt.colorbar()
plt.show()
