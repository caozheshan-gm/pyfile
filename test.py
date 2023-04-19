import numpy as np

# Sample data
X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
y = np.array([0, 0, 0, 1, 1, 1])

num_features = X.shape[1]
num_classes = len(np.unique(y))

# Calculate the mean of each class
class_means = []
for i in range(num_classes):
    class_mean = np.mean(X[y == i], axis=0)
    class_means.append(class_mean)

# Calculate the overall mean
overall_mean = np.mean(X, axis=0)

# Calculate the within-class scatter matrix (Sw)
Sw = np.zeros((num_features, num_features))
for i in range(num_classes):
    class_scatter = np.zeros((num_features, num_features))
    for x in X[y == i]:
        x = x.reshape(-1, 1)
        class_mean = class_means[i].reshape(-1, 1)
        class_scatter += (x - class_mean) @ (x - class_mean).T
    Sw += class_scatter

# Calculate the between-class scatter matrix (Sb)
Sb = np.zeros((num_features, num_features))
for i in range(num_classes):
    n = len(X[y == i])
    class_mean = class_means[i].reshape(-1, 1)
    overall_mean = overall_mean.reshape(-1, 1)
    Sb += n * (class_mean - overall_mean) @ (class_mean - overall_mean).T

print(overall_mean)