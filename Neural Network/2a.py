import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize weights
np.random.seed(42)
def initialize_weights(input_size, hidden1_size, hidden2_size, output_size):
    weights = {
        'hidden1': np.random.randn(input_size, hidden1_size) * 0.01,
        'hidden2': np.random.randn(hidden1_size, hidden2_size) * 0.01,
        'output': np.random.randn(hidden2_size, output_size) * 0.01
    }
    return weights

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def forward_pass(X, weights):
    hidden1_input = np.dot(X, weights['hidden1'])
    hidden1_output = sigmoid_function(hidden1_input)

    hidden2_input = np.dot(hidden1_output, weights['hidden2'])
    hidden2_output = sigmoid_function(hidden2_input)

    output_input = np.dot(hidden2_output, weights['output'])
    output = sigmoid_function(output_input)

    return hidden1_output, hidden2_output, output

def backward_pass(X, y, weights, hidden1_output, hidden2_output, output, learning_rate):
    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden2_error = np.dot(output_delta, weights['output'].T)
    hidden2_delta = hidden2_error * sigmoid_derivative(hidden2_output)

    hidden1_error = np.dot(hidden2_delta, weights['hidden2'].T)
    hidden1_delta = hidden1_error * sigmoid_derivative(hidden1_output)

    weights['output'] += np.outer(hidden2_output, output_delta) * learning_rate
    weights['hidden2'] += np.outer(hidden1_output, hidden2_delta) * learning_rate
    weights['hidden1'] += np.outer(X, hidden1_delta) * learning_rate

def train_network(X_train, y_train, weights, learning_rate, epochs):
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X_train)):
            X = X_train[i]
            y = y_train[i]
            hidden1_output, hidden2_output, output = forward_pass(X, weights)
            backward_pass(X, y, weights, hidden1_output, hidden2_output, output, learning_rate)
            epoch_loss += np.mean((y - output) ** 2)

def predict_network(X_test, weights):
    predictions = []
    for i in range(len(X_test)):
        _, _, output = forward_pass(X_test[i], weights)
        predictions.append(1 if output[0] >= 0.5 else 0)
    return np.array(predictions)

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

# Initialize and train neural network
input_size = X_train.shape[1]
hidden1_size = 8
hidden2_size = 8
output_size = 1
learning_rate = 0.01
epochs = 50

weights = initialize_weights(input_size, hidden1_size, hidden2_size, output_size)
train_network(X_train, y_train, weights, learning_rate, epochs)

# Test neural network
predictions = predict_network(X_test, weights)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")