import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from itertools import product

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / gamma)

def train_kernel_perceptron(X, y, gamma, max_iter):
    n = len(y)
    alpha = np.zeros(n)
    for iteration in range(max_iter):
        for i in range(n):
            prediction = np.sign(
                np.sum(
                    alpha * y *
                    np.array([gaussian_kernel(X[j], X[i], gamma) for j in range(n)])
                )
            )
            if prediction != y[i]:
                alpha[i] += 1
    return alpha

def predict_kernel_perceptron(X_train, y_train, alpha, X_test, gamma):
    predictions = []
    for x in X_test:
        prediction = np.sign(
            np.sum(
                alpha * y_train *
                np.array([gaussian_kernel(sv, x, gamma) for sv in X_train])
            )
        )
        predictions.append(prediction)
    return np.array(predictions)

def train_kernel_svm(X, y, C, gamma):
    n, d = X.shape
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)

    def dual_objective(alpha):
        return 0.5 * np.sum((alpha[:, None] * y[:, None]) @ (alpha[None, :] * y[None, :]) * K) - np.sum(alpha)

    constraints = [{'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}]
    bounds = [(0, C) for _ in range(n)]
    alpha_init = np.zeros(n)
    result = minimize(dual_objective, alpha_init, bounds=bounds, constraints=constraints, method='SLSQP')
    alpha = result.x
    support_indices = (alpha > 1e-5)
    support_vectors = X[support_indices]
    support_labels = y[support_indices]
    support_alpha = alpha[support_indices]
    b = np.mean([
        support_labels[i] - np.sum(support_alpha * support_labels * K[support_indices][:, i])
        for i in range(len(support_alpha))
    ])
    return support_alpha, support_vectors, support_labels, b

def predict_kernel_svm(X, support_alpha, support_vectors, support_labels, b, gamma):
    predictions = []
    for x in X:
        prediction = np.sum(
            support_alpha * support_labels *
            [gaussian_kernel(x, sv, gamma) for sv in support_vectors]
        ) + b
        predictions.append(np.sign(prediction))
    return np.array(predictions)

def main():
    train_data = pd.read_csv(r"bank-note\train.csv", header=None)
    test_data = pd.read_csv(r"bank-note\test.csv", header=None)
    train_data.iloc[:, -1] = train_data.iloc[:, -1].map({1: 1, 0: -1})
    test_data.iloc[:, -1] = test_data.iloc[:, -1].map({1: 1, 0: -1})

    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

    gamma_values = [0.1, 0.5, 1, 5, 100]
    C_values = ["100 / 873", "500 / 873", "700 / 873"]

    print("\nKernel Perceptron Results:")
    for gamma in gamma_values:
        print(f"Training Kernel Perceptron with gamma = {gamma}")
        alpha = train_kernel_perceptron(X_train, y_train, gamma, max_iter=100)
        train_preds = predict_kernel_perceptron(X_train, y_train, alpha, X_train, gamma)
        test_preds = predict_kernel_perceptron(X_train, y_train, alpha, X_test, gamma)
        train_error = 1 - accuracy_score(y_train, train_preds)
        test_error = 1 - accuracy_score(y_test, test_preds)
        print(f"Gamma = {gamma}, Train Error = {train_error:.4f}, Test Error = {test_error:.4f}")

    print("\nKernel SVM Results:")
    for C_str, gamma in product(C_values, gamma_values):
        print(f"Training Kernel SVM with C = {C_str}, gamma = {gamma}")
        C = eval(C_str)
        support_alpha, support_vectors, support_labels, b = train_kernel_svm(X_train, y_train, C, gamma)
        train_preds = predict_kernel_svm(X_train, support_alpha, support_vectors, support_labels, b, gamma)
        test_preds = predict_kernel_svm(X_test, support_alpha, support_vectors, support_labels, b, gamma)
        train_error = 1 - accuracy_score(y_train, train_preds)
        test_error = 1 - accuracy_score(y_test, test_preds)
        print(f"C = {C_str}, Gamma = {gamma}, Train Error = {train_error:.4f}, Test Error = {test_error:.4f}")

if __name__ == "__main__":
    main()