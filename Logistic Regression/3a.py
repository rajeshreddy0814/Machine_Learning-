import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(X, y, weights, variance):
    predictions = sigmoid(np.dot(X, weights))
    gradient = -np.dot(X.T, (y - predictions)) + (weights / variance)
    return gradient

def compute_loss(X, y, weights, variance):
    predictions = sigmoid(np.dot(X, weights))
    likelihood = y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9)
    prior = -np.sum(weights**2) / (2 * variance)
    return -np.mean(likelihood) + prior

def train_model(X, y, weights, variance, gamma_0, d, epochs):
    for epoch in range(epochs):
        permutation = np.random.permutation(len(X))
        X = X[permutation]
        y = y[permutation]
        for t in range(len(X)):
            learning_rate = gamma_0 / (1 + (gamma_0 / d) * (epoch * len(X) + t))
            gradient = compute_gradient(X[t:t+1], y[t:t+1], weights, variance)
            weights -= learning_rate * gradient
    return weights

def predict_model(X, weights):
    return (sigmoid(np.dot(X, weights)) >= 0.5).astype(int)

# Load and preprocess data
train_data = pd.read_csv("datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("datasets/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
gamma_0 = 0.5
d = 1e-2
epochs = 100

print("Variance | Train Error | Test Error")
print("-----------------------------------")
for variance in variances:
    weights = np.zeros(X_train.shape[1])
    weights = train_model(X_train, y_train, weights, variance, gamma_0, d, epochs)

    train_predictions = predict_model(X_train, weights)
    test_predictions = predict_model(X_test, weights)

    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)

    print(f"{variance:.2f}     | {train_error:.4f}    | {test_error:.4f}")
