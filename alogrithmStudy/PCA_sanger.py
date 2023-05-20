import numpy as np

def gha(data, n_components, learning_rate=0.001, n_epochs=1000):
    n_samples, n_features = data.shape

    # Initialize the weight matrix with random values
    W = np.random.randn(n_features, n_components)

    # Iterate through the dataset
    for epoch in range(n_epochs):
        for i in range(n_samples):
            x = data[i, :]

            # Compute the output vector y = W^T * x
            y = np.dot(W.T, x)

            # Update the weight matrix W using Sanger's rule
            delta_W = np.outer(x, y) - np.dot(W, np.tril(np.outer(y, y)))
            W += learning_rate * delta_W

    return W

# Generate some sample data (e.g., 100 samples, 5 features)
data = np.random.randn(100, 5)

# Apply GHA to reduce the data dimension to 2
W = gha(data, n_components=2)

# Project the data onto the new subspace
reduced_data = np.dot(data, W)