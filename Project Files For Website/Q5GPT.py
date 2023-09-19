import numpy as np
from sklearn.datasets import make_moons
from scipy.cluster.hierarchy import fcluster, linkage, cut_tree
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt


np.random.seed(42)

# First moon
n_samples = 100
X1, y1 = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
X1[:, 0] = X1[:, 0] * 3 - 1 + 2
X1[:, 1] = X1[:, 1] * 3 + 2
y1 += 1

# Second moon
X2, y2 = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
X2[:, 0] = -X2[:, 0] * 2.5 + 6 + 3
X2[:, 1] = X2[:, 1] * 2.5 + 1 + 2
X2 = np.dot(X2 - [3, 2], [[0, 1], [-1, 0]]) + [3, 2]
y2 += 2

# Third group
X3 = np.zeros((n_samples, 2))
X3[:, 0] = np.random.normal(loc=2, scale=0.1, size=n_samples)
X3[:, 1] = np.random.normal(loc=5, scale=4, size=n_samples)
y3 = np.full(n_samples, 3)

# Fourth group
theta = np.random.uniform(0, 2*np.pi, n_samples)
r = 2 * np.sqrt(np.random.uniform(0, 1, n_samples))
X4 = np.zeros((n_samples, 2))
X4[:, 0] = r * np.cos(theta) + 3
X4[:, 1] = r * np.sin(theta) + 3
y4 = np.full(n_samples, 4)

# Fifth group
X5 = np.random.uniform(-2.5, 2.5, size=(n_samples, 2))
X5 += [0, 0]
y5 = np.full(n_samples, 5)

X = np.concatenate((X1, X2, X3, X4, X5), axis=0)
y = np.concatenate((y1, y2, y3, y4, y5), axis=0)

####################################################################
## CURE Algorithm ##

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Z = linkage(X_scaled, method='ward')
# k = 5
# labels = fcluster(Z, k, criterion='maxclust')

# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
# plt.title("CURE clustering")
# plt.show()
########################################################################

####################################################################
## spectral clustering ##

# sc = SpectralClustering(n_clusters=5)
# sc.fit(X)

# plt.scatter(X[:, 0], X[:, 1], c=sc.labels_, cmap='viridis')
# plt.title("Spectral Clustering Results")
# plt.show()
########################################################################

####################################################################
## SNN Algorithm ##

# Compute shared nearest neighbor graph
# connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
# shared_nn_graph = (connectivity + connectivity.T).toarray()

# # Compute minimum spanning tree
# mst = minimum_spanning_tree(shared_nn_graph)

# # Perform hierarchical clustering on the minimum spanning tree
# Z = linkage(mst.toarray(), method='complete')

# # Perform SNN clustering with k=5
# k = 5
# clusters = fcluster(Z, k, criterion='maxclust')

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='rainbow')
# plt.title('SNN Clustering Results (k=5)')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()
########################################################################

####################################################################
## Chameleon Algorithm ##

# Define the distance function
def dist(a, b):
    return np.linalg.norm(a - b)

# Combine all samples into one array
samples = np.concatenate((X1, X2, X3, X4, X5), axis=0)

# Normalize the samples
scaler = StandardScaler()
samples = scaler.fit_transform(samples)

# Compute the distance matrix
n_samples = samples.shape[0]
dist_matrix = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(i+1, n_samples):
        d = dist(samples[i], samples[j])
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

# Compute the minimum spanning tree
mst = minimum_spanning_tree(dist_matrix)

# Compute the distances between each point and its k-nearest neighbors
k = 5
graph = kneighbors_graph(samples, k, mode='distance')

# Merge the minimum spanning tree and the k-nearest neighbor graph
merged_graph = graph.maximum(mst)

# Compute the pairwise distances between the samples
A = kneighbors_graph(X, n_neighbors=10, metric='euclidean', mode='distance', include_self=False)
D = minimum_spanning_tree(A)
D = D.toarray()

# Symmetrize the distance matrix
D = 0.5 * (D + D.T)

# Run the Chameleon algorithm
k = 5
L = linkage(squareform(D), method='complete')
T = cut_tree(L, n_clusters=k)

# Cut the dendrogram at the desired number of clusters
num_clusters = 5
labels = cut_tree(L, n_clusters=num_clusters)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
########################################################################
