import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize

def preprocess_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data.iloc[:, -1] = data.iloc[:, -1].map({1: 1, 0: -1})
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def train_primal(X, y, C, gamma0, max_epochs):
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    updates = 0
    for epoch in range(max_epochs):
        X, y = shuffle(X, y, random_state=epoch)
        for i in range(n):
            updates += 1
            eta = gamma0 / (1 + (gamma0 / 0.01) * updates)
            margin = y[i] * (np.dot(X[i], w) + b)
            if margin < 1:
                w = (1 - eta) * w + eta * C * y[i] * X[i]
                b += eta * C * y[i]
            else:
                w = (1 - eta) * w
    return w, b

def train_dual(X, y, C):
    n, d = X.shape

    def dual_objective(alpha):
        return 0.5 * np.sum((alpha[:, None] * y[:, None] * X).sum(axis=0) ** 2) - np.sum(alpha)

    constraints = [{'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}]
    bounds = [(0, C) for _ in range(n)]
    alpha_init = np.zeros(n)

    result = minimize(dual_objective, alpha_init, bounds=bounds, constraints=constraints, method='SLSQP')
    alpha_optimal = result.x

    w = np.dot((alpha_optimal * y)[:, None].T, X).flatten()
    support_vectors = (alpha_optimal > 1e-5) & (alpha_optimal < C - 1e-5)
    b = np.mean(y[support_vectors] - np.dot(X[support_vectors], w))
    return w, b

def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

def evaluate(X, y, w, b):
    predictions = predict(X, w, b)
    error_rate = np.mean(predictions != y)
    return error_rate

def main():
    train_file = r"bank-note\train.csv"
    test_file = r"bank-note\test.csv"
    X_train, y_train = preprocess_data(train_file)
    X_test, y_test = preprocess_data(test_file)

    C_values = ["100/873", "500/873", "700/873"]
    gamma0 = 0.1
    max_epochs = 100

    for C in C_values:
        print(f"Training with C = {C}")

        numerical_C = eval(C)
        w_primal, b_primal = train_primal(X_train, y_train, numerical_C, gamma0, max_epochs)
        train_error_primal = evaluate(X_train, y_train, w_primal, b_primal)
        test_error_primal = evaluate(X_test, y_test, w_primal, b_primal)
        print("Primal SVM:")
        print(f"  Weights (w): {w_primal}")
        print(f"  Bias (b): {b_primal}")
        print(f"  Training Error: {train_error_primal}")
        print(f"  Testing Error: {test_error_primal}")

        w_dual, b_dual = train_dual(X_train, y_train, numerical_C)
        train_error_dual = evaluate(X_train, y_train, w_dual, b_dual)
        test_error_dual = evaluate(X_test, y_test, w_dual, b_dual)
        print("Dual SVM:")
        print(f"  Weights (w): {w_dual}")
        print(f"  Bias (b): {b_dual}")
        print(f"  Training Error: {train_error_dual}")
        print(f"  Testing Error: {test_error_dual}")

        weight_difference = np.linalg.norm(w_primal - w_dual)
        bias_difference = abs(b_primal - b_dual)
        print("Comparison:")
        print(f"  Weight Difference: {weight_difference}")
        print(f"  Bias Difference: {bias_difference}")
        print("=" * 30)

if __name__ == "__main__":
    main()
