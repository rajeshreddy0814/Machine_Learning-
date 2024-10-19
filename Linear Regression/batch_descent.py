#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to implement batch gradient descent
def perform_gradient_descent(X, y, lr_init=1.0, tolerance=1e-6, max_iter=1000):
    X = np.c_[np.ones(X.shape[0]), X]  # Add intercept term (bias)
    weights = np.zeros(X.shape[1])  # Initialize weights to zeros
    lr = lr_init
    cost_history = []

    for i in range(max_iter):
        predictions = X.dot(weights)
        residuals = predictions - y
        grad = X.T.dot(residuals) / len(y)
        updated_weights = weights - lr * grad

        # Compute the weight difference norm
        weight_diff_norm = np.linalg.norm(updated_weights - weights)

        # Calculate and record cost function value (MSE)
        cost = np.mean(residuals ** 2)
        cost_history.append(cost)

        # Check convergence condition
        if weight_diff_norm < tolerance:
            break

        weights = updated_weights

        # Adjust learning rate if necessary
        if lr > 0.001 and weight_diff_norm > tolerance:
            lr *= 0.5

    return weights, cost_history, lr

# Main method to execute the program
def main():
    # Loading the training and test datasets
    train = pd.read_csv("data/concrete/train.csv")
    test = pd.read_csv("data/concrete/test.csv")

    # Separating features and target variables for training and testing sets
    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # Running the gradient descent function
    final_weights, costs, learning_rate_used = perform_gradient_descent(X_train, y_train)

    # Plotting the evolution of the cost function over iterations
    plt.figure(figsize=(10, 6))
    plt.plot(costs, marker='o', linestyle='-', color='red', markersize=4)
    plt.title('Cost Function Over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()

    # Evaluating on test data
    X_test_with_bias = np.c_[np.ones(X_test.shape[0]), X_test]
    test_preds = X_test_with_bias.dot(final_weights)
    mse_test = np.mean((test_preds - y_test) ** 2)

    # Output the results
    print("Final Learning Rate:", learning_rate_used)
    print("Test Data MSE:", mse_test)
    print("Final Model Weights:", final_weights)

# Ensure the script runs when executed
if __name__ == "__main__":
    main()

