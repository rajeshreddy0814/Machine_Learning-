import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(X, y, weights):
    predictions = sigmoid(np.dot(X, weights))
    gradient = -np.dot(X.T, (y - predictions))
    return gradient

def compute_loss(X, y, weights):
    predictions = sigmoid(np.dot(X, weights))
    likelihood = y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9)
    return -np.mean(likelihood)

def train(X, y, gamma_0, d, epochs, input_size):
    weights = np.zeros(input_size)
    losses = []
    for epoch in range(epochs):
        permutation = np.random.permutation(len(X))
        X = X[permutation]
        y = y[permutation]
        for t in range(len(X)):
            learning_rate = gamma_0 / (1 + (gamma_0 / d) * (epoch * len(X) + t))
            gradient = compute_gradient(X[t:t+1], y[t:t+1], weights)
            weights -= learning_rate * gradient
        loss = compute_loss(X, y, weights)
        losses.append(loss)
    return weights, losses

def predict(X, weights):
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

results = []

for variance in variances:
    print(f"\nTraining with variance: {variance}")
    weights, losses = train(X_train, y_train, gamma_0, d, epochs, X_train.shape[1])

    train_predictions = predict(X_train, weights)
    test_predictions = predict(X_test, weights)

    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)

    results.append((variance, train_error, test_error))

    print(f"Train Error: {train_error:.4f} | Test Error: {test_error:.4f}")

# Report results
print("\nResults:")
print("Variance\tTrain Error\tTest Error")
for result in results:
    print(f"{result[0]}\t\t{result[1]:.4f}\t\t{result[2]:.4f}")