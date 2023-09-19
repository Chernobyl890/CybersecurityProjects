import numpy as np
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=250, centers=3, random_state=42)

def dot(X, Y):
    result = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(X.shape[1]):
                result[i][j] += X[i][k] * Y[k][j]
    return result

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def fcm(X, n_clusters=3, max_iter=100, m=2, error=1e-5, random_state=42):
    np.random.seed(random_state)
    
    # Initialize membership matrix
    U = np.random.rand(X.shape[0], n_clusters)
    U = np.divide(U, np.sum(U, axis=1)[:, np.newaxis])
    
    # Initialize centroid matrix
    C = np.zeros((n_clusters, X.shape[1]))
    for j in range(n_clusters):
        C[j] = np.sum(U[:, j][:, np.newaxis] * X, axis=0) / np.sum(U[:, j])
    
    # Repeat until convergence
    for i in range(max_iter):
        
        # Update membership matrix
        U_old = np.copy(U)
        for j in range(n_clusters):
            for k in range(X.shape[0]):
                den = 0
                for l in range(n_clusters):
                    den += (euclidean_distance(X[k], C[j]) / euclidean_distance(X[k], C[l])) ** (2 / (m - 1))
                U[k][j] = 1 / den
        
        # Check for convergence
        if np.max(np.abs(U - U_old)) < error:
            break
        
        # Update centroid matrix
        for j in range(n_clusters):
            num = np.zeros(X.shape[1])
            den = 0
            for k in range(X.shape[0]):
                num += (U[k][j] ** m) * X[k]
                for l in range(n_clusters):
                    den += (U[k][l] ** m)
            C[j] = num / den
    
    # Assign clusters
    labels = np.argmax(U, axis=1)
    
    return labels, C

# Add some noise and outliers
X[:10] += np.random.normal(scale=10, size=(10, 2))
X[200:] += np.random.normal(scale=10, size=(50, 2))

# Cluster data using FCM
labels, centroids = fcm(X)

# Plot results
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.title('FCM Clustering with Noise and Outliers')
plt.show()
