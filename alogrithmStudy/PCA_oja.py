import numpy as np

# Sample data matrix
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Initialize the weight matrix
W = np.random.randn(X.shape[1], 2)

# Set the learning rate
learning_rate = 0.1

# Train the network using Hebbian learning
for i in range(X.shape[0]):
    # Forward pass
    y = np.dot(X[i], W)
    
    # Backward pass (Hebbian learning rule)
    W += learning_rate * np.outer(X[i]-np.dot(y,W.T), y)
# Project the data onto the learned principal components
Z = np.dot(X, W)

# Compare the dimensionality-reduced data to the original data
print("Original data:\n", X)
print("Dimensionality-reduced data:\n", Z)