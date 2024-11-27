import numpy as np
import pandas as pd

def preprocess_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    y = np.where(y == 0, -1, 1)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X, y

def svm_train(X, y, C, gamma_0, a, max_epochs):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for epoch in range(max_epochs):
        indices = np.random.permutation(n_samples)
        for t, i in enumerate(indices):
            learning_rate = gamma_0 / (1 + (gamma_0 / a) * (epoch * n_samples + t))
            condition = y[i] * (np.dot(weights, X[i]) + bias)
            if condition <= 1:
                weights = (1 - learning_rate) * weights + learning_rate * C * y[i] * X[i]
                bias += learning_rate * C * y[i]
            else:
                weights = (1 - learning_rate) * weights

    return weights, bias

def svm_predict(X, weights, bias):
    return np.sign(np.dot(X, weights) + bias)

def evaluate(X, y, weights, bias):
    predictions = svm_predict(X, weights, bias)
    error_rate = np.mean(predictions != y)
    return error_rate

def main():
    train_file = r"bank-note\train.csv"
    test_file =r"bank-note\test.csv"

    X_train, y_train = preprocess_data(train_file)
    X_test, y_test = preprocess_data(test_file)

    C_values = [("100/873", 100 / 873), ("500/873", 500 / 873), ("700/873", 700 / 873)]
    gamma_0 = 0.01
    a = 0.01
    max_epochs = 100

    for C_label, C_value in C_values:
        print(f"\nTraining SVM with C = {C_label}")
        weights, bias = svm_train(X_train, y_train, C_value, gamma_0, a, max_epochs)
        train_error = evaluate(X_train, y_train, weights, bias)
        test_error = evaluate(X_test, y_test, weights, bias)
        print(f"Training error: {train_error:.4f}")
        print(f"Test error: {test_error:.4f}")

if __name__ == "__main__":
    main()
