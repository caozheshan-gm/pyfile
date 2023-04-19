import numpy as np

# Oja's Rule
def ojas_rule(data, learning_rate=0.001, n_epochs=1000):
    n_samples, n_features = data.shape

    # Initialize the weight vector with random values
    w = np.random.randn(n_features)

    # Iterate through the dataset
    for epoch in range(n_epochs):
        for i in range(n_samples):
            x = data[i, :]
            y = np.dot(w.T, x)

            # Update the weight vector using Oja's rule
            w += learning_rate * y * (x - y * w)

    return w

# Oja's Subspace Rule
def ojas_subspace_rule(data, n_components, learning_rate=0.001, n_epochs=1000):
    n_samples, n_features = data.shape

    # Initialize the weight matrix with random values
    W = np.random.randn(n_features, n_components)

    # Iterate through the dataset
    for epoch in range(n_epochs):
        for i in range(n_samples):
            x = data[i, :]
            y = np.dot(W.T, x)

            # Update the weight matrix using Oja's subspace rule
            delta_W = np.outer(x, y) - np.dot(W, np.tril(np.outer(y, y)))
            W += learning_rate * delta_W

    return W

# Generate some sample data (e.g., 100 samples, 5 features)
data = np.random.randn(100, 5)

# Apply Oja's Rule to find the first principal component
w = ojas_rule(data)

# Apply Oja's Subspace Rule to find the first 2 principal components
W = ojas_subspace_rule(data, n_components=2)