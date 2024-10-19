#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

# Load train and test data
train_data = pd.read_csv("data/concrete/train.csv")
test_data = pd.read_csv("data/concrete/test.csv")

# Separate features and target for training and testing
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# Function to compute optimal weights using the normal equation
def compute_optimal_weights(X, y):
    # Add a bias term (column of ones) to the feature matrix X
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Compute the optimal weights using the normal equation
    optimal_weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    return optimal_weights

# Function to calculate the cost (Mean Squared Error)
def compute_cost(X, y, weights):
    # Add a bias term to the feature matrix
    X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Calculate the predictions
    predictions = X_bias @ weights
    
    # Compute the mean squared error
    m = len(y)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Calculate the optimal weights using the training data
optimal_weights = compute_optimal_weights(X_train, y_train)

test_cost = compute_cost(X_test, y_test, optimal_weights)

print("Optimal Weights:", optimal_weights)
print("Cost:", test_cost)

