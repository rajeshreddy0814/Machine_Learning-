import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

# Node and DecisionTree classes remain unchanged (from your code)

class Node:
    def _init_(self, feature=None, value=None, left=None, right=None, is_numerical=False):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.prediction = None
        self.is_numerical = is_numerical

class DecisionTree:
    def _init_(self, max_depth=None, criterion='information_gain'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            leaf = Node(value=self._most_common_label(y))
            leaf.prediction = leaf.value
            return leaf

        best_feature, best_value, is_numerical = self._best_split(X, y)
        if best_feature is None:
            leaf = Node(value=self._most_common_label(y))
            leaf.prediction = leaf.value
            return leaf

        if is_numerical:
            left_indices = X[:, best_feature] <= best_value
            right_indices = X[:, best_feature] > best_value
        else:
            left_indices = X[:, best_feature] == best_value
            right_indices = X[:, best_feature] != best_value

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, value=best_value, left=left_child, right=right_child, is_numerical=is_numerical)

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_value = None
        best_is_numerical = False

        for feature in range(X.shape[1]):
            values = X[:, feature]
            if np.issubdtype(values.dtype, np.number):
                sorted_values = np.unique(values)
                thresholds = (sorted_values[:-1] + sorted_values[1:]) / 2
                is_numerical = True

                for threshold in thresholds:
                    left_indices = values <= threshold
                    right_indices = values > threshold
                    if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                        continue

                    gain = self._calculate_gain(y, y[left_indices], y[right_indices])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_value = threshold
                        best_is_numerical = is_numerical
            else:
                unique_values = set(values)
                is_numerical = False

                for value in unique_values:
                    left_indices = values == value
                    right_indices = values != value
                    if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                        continue

                    gain = self._calculate_gain(y, y[left_indices], y[right_indices])
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_value = value
                        best_is_numerical = is_numerical

        return best_feature, best_value, best_is_numerical

    def _calculate_gain(self, y, left_y, right_y):
        if self.criterion == 'information_gain':
            return self._information_gain(y, left_y, right_y)

    def _information_gain(self, y, left_y, right_y):
        parent_entropy = self._entropy(y)
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        child_entropy = (n_left / n) * self._entropy(left_y) + (n_right / n) * self._entropy(right_y)
        return parent_entropy - child_entropy

    def _entropy(self, y):
        counter = Counter(y)
        probabilities = np.array(list(counter.values())) / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_instance(x) for x in X])

    def _predict_instance(self, x):
        node = self.root
        while node.left or node.right:
            if node.is_numerical:
                if x[node.feature] <= node.value:
                    node = node.left
                else:
                    node = node.right
            else:
                if x[node.feature] == node.value:
                    node = node.left
                else:
                    node = node.right
        return node.prediction

# BaggedTrees class
class BaggedTrees:
    def _init_(self):
        self.trees = []

    def add_tree(self, X, y):
        n_samples = X.shape[0]
        X_sample, y_sample = resample(X, y, n_samples=n_samples, replace=True)
        tree = DecisionTree(max_depth=None)  # Fully grown trees
        tree.fit(X_sample, y_sample)
        self.trees.append(tree)

    def predict(self, X):
        # Get predictions from all trees and take the majority vote
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        majority_votes = [Counter(tree_predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(majority_votes)

# Bias and variance calculation function
def calculate_bias_variance(bagged_preds, single_tree_preds, y_test):
    bias_sq_single = np.mean((np.mean(single_tree_preds, axis=0) - y_test) ** 2)
    bias_sq_bagged = np.mean((np.mean(bagged_preds, axis=0) - y_test) ** 2)

    var_single = np.mean(np.var(single_tree_preds, axis=0))
    var_bagged = np.mean(np.var(bagged_preds, axis=0))

    return bias_sq_single, var_single, bias_sq_bagged, var_bagged

# Experiment with Bias and Variance (Add trees incrementally)
def experiment_with_bias_variance(X_train, y_train, X_test, y_test, n_repeats=100, n_trees=50):
    bagged_preds = []
    single_tree_preds = []

    for _ in range(n_repeats):
        X_sample, y_sample = resample(X_train, y_train, n_samples=1000, replace=False)

        # Incrementally add trees
        bagged_trees = BaggedTrees()

        # Store predictions for both bagged trees and a single tree
        bagged_pred_instance = []
        single_tree_pred_instance = []

        for t in range(n_trees):
            bagged_trees.add_tree(X_sample, y_sample)

            # Collect bagged trees predictions
            if t == 0:
                # Save the first tree's prediction (for single tree analysis)
                single_tree_pred_instance.append(bagged_trees.trees[0].predict(X_test))

            bagged_pred_instance.append(bagged_trees.predict(X_test))

        # Save predictions across repeats
        bagged_preds.append(bagged_pred_instance[-1])  # Final bagged prediction
        single_tree_preds.append(single_tree_pred_instance[0])  # First tree prediction

    # Convert predictions to numpy arrays for bias and variance calculation
    bagged_preds = np.array(bagged_preds)
    single_tree_preds = np.array(single_tree_preds)

    # Calculate bias and variance
    bias_sq_single, var_single, bias_sq_bagged, var_bagged = calculate_bias_variance(bagged_preds, single_tree_preds, y_test)

    print(f"Single Tree - Bias^2: {bias_sq_single:.4f}, Variance: {var_single:.4f}, Squared Error: {bias_sq_single + var_single:.4f}")
    print(f"Bagged Trees - Bias^2: {bias_sq_bagged:.4f}, Variance: {var_bagged:.4f}, Squared Error: {bias_sq_bagged + var_bagged:.4f}")

# Load datasets
column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
                  'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

df_train = pd.read_csv("data/bank/train.csv", names=column_headers)
df_test = pd.read_csv("data/bank/test.csv", names=column_headers)

y_train = df_train['label'].apply(lambda x: 1 if x == 'yes' else -1).values
y_test = df_test['label'].apply(lambda x: 1 if x == 'yes' else -1).values

X_train = df_train.drop('label', axis=1).values
X_test = df_test.drop('label', axis=1).values

# Run the experiment
experiment_with_bias_variance(X_train, y_train, X_test, y_test)