import numpy as np

# Example data
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12],
              [13, 14, 15],])

# Compute the covariance matrix
mean_X = np.mean(X, axis=0)
X_centered = X - mean_X
cov_matrix = np.cov(X_centered, rowvar=False)

# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Select the top k eigenvectors
k = 3  # In this case, k equals the number of features
top_k_indices = np.argsort(eigenvalues)[-k:]
top_k_eigenvectors = eigenvectors[:, top_k_indices]

# Form the projection matrix (W)
W = top_k_eigenvectors

# Project the data
Y = X_centered.dot(W)

print(W)