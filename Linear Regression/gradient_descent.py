#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Function for stochastic gradient descent algorithm
def sgd(X, y, lr_initial=0.01, max_epochs=50, tol=1e-6):
    np.random.seed(0)  # Set random seed for reproducibility
    X = np.c_[np.ones(X.shape[0]), X]  # Add bias term (intercept)
    weights = np.zeros(X.shape[1])  # Initialize weights as zero
    lr = lr_initial
    cost_history = []
    prev_cost = float('inf')

    # Iterate through each epoch
    for epoch in range(max_epochs):
        for _ in range(len(y)):
            # Select a random sample from the dataset
            idx = np.random.randint(0, len(y))
            x_i = X[idx]
            y_i = y[idx]

            # Make prediction and compute error
            pred = np.dot(x_i, weights)
            error = pred - y_i
            gradient = x_i * error  # Calculate gradient for the sample

            # Update weights using gradient
            weights -= lr * gradient

            # Compute cost for the entire dataset
            all_preds = np.dot(X, weights)
            cost = np.mean((all_preds - y) ** 2)
            cost_history.append(cost)

            # Check for convergence
            if abs(prev_cost - cost) < tol:
                return weights, cost_history, lr
            prev_cost = cost

        # Decay the learning rate over epochs
        lr /= 1.02

    return weights, cost_history, lr

# Main function to run the code
def main():
    # Load the training and test datasets
    train_data = pd.read_csv("data/concrete/train.csv")
    test_data = pd.read_csv("data/concrete/test.csv")

    # Separate features and target for training and testing
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    # Running the SGD algorithm
    final_weights, cost_history, final_lr = sgd(X_train, y_train)

    # Plotting cost function vs updates
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, linestyle='-', marker='o', markersize=4, color='red')
    plt.title('Cost Function vs. Updates')
    plt.xlabel('Number of Updates')
    plt.ylabel('Cost Function Value')
    plt.grid(True)
    plt.show()

    # Evaluate the model on test data
    X_test_with_bias = np.c_[np.ones(X_test.shape[0]), X_test]
    test_preds = np.dot(X_test_with_bias, final_weights)
    test_mse = np.mean((test_preds - y_test) ** 2)

    # Output results
    print("Final learning rate:", final_lr)
    print("Final weights:", final_weights)
    print("Test data cost (MSE):", test_mse)

# Ensures the script runs only when executed directly
if __name__ == "__main__":
    main()

