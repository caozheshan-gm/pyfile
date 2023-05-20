import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
n_samples = 1000
time = np.linspace(0, 8, n_samples)
source1 = np.sin(3 * time)
source2 = np.sign(np.sin(4 * time))
sources = np.c_[source1, source2]

# Create a mixing matrix and mix the sources
mixing_matrix = np.array([[0.5, 0.5], [0.5, -0.5]])
mixed_signals = np.dot(sources, mixing_matrix.T)

# Perform PCA using Oja's subspace rule for data whitening
def oja_subspace_rule(X, n_components, n_iterations, learning_rate):
    n_features = X.shape[1]
    W = np.random.rand(n_components, n_features)

    for _ in range(n_iterations):
        for x in X:
            y = W @ x
            delta_W = learning_rate * (np.outer(y, x) - (y ** 2).reshape(-1, 1) * W)
            W += delta_W

    return W

n_components = 2
n_iterations = 1000
learning_rate = 0.001
W_pca = oja_subspace_rule(mixed_signals, n_components, n_iterations, learning_rate)
whitened_data = mixed_signals @ W_pca.T

# Scale the whitened data to have unit variance
scaler = StandardScaler(with_mean=False)
whitened_data = scaler.fit_transform(whitened_data)

# Perform ICA using FastICA
ica = FastICA(n_components=2)
recovered_signals = ica.fit_transform(whitened_data)