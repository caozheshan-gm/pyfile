import numpy as np

# Dataset
X = np.array([[0.5, 1.0],
              [1.5, 2.0],
              [2.0, 1.5],
              [1.0, 0.5]])

T = np.array([[0],
              [0],
              [1],
              [1]])

# Initialize input weights and biases
np.random.seed(42)  # for reproducibility
n_input = 2
n_hidden = 4

W = np.random.randn(n_input, n_hidden)
b = np.random.randn(1, n_hidden)

# Activation function: sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Calculate hidden layer output matrix H
H = sigmoid(X @ W + b)

# Compute output weights Beta using Moore-Penrose pseudo-inverse
Beta = np.linalg.pinv(H) @ T

# Prediction function
def predict(X_new):
    H_new = sigmoid(X_new @ W + b)
    y_new = H_new @ Beta
    return (y_new > 0.5).astype(int)

# Example prediction on a new input
X_new = np.array([[1.2, 0.8]])
y_new = predict(X_new)

print(f"Prediction for input {X_new}: {y_new[0, 0]}")