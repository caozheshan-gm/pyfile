import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the Iris dataset
data = load_iris()
X = data['data']
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the LDA object and fit the model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Transform the data to a lower-dimensional space
X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

# Use the LDA model to make predictions on the test set
y_pred = lda.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=data['target_names']))