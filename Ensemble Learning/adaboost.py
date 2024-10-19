import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class Stump:
    def __init__(self, criterion='info_gain'):
        self.max_depth = 1
        self.criterion = criterion
        self.root = None

    def fit(self, X, y, weights=None):
        self.n_features = X.shape[1]
        self.feature_types = self._determine_types(X)
        self.root = self._grow_tree(X, y, weights)

    def _determine_types(self, X):
        types = []
        for i in range(X.shape[1]):
            unique_values = np.unique(X[:, i])
            if isinstance(unique_values[0], (int, float)) and len(unique_values) > 10:
                types.append("num")
            else:
                types.append("cat")
        return types

    def _grow_tree(self, X, y, weights):
        best_feat, best_value = self._best_split(X, y, weights)
        if best_feat is None:
            return Node(value=self._common_label(y, weights))
        left_idx, right_idx = self._split(X, best_feat, best_value)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return Node(value=self._common_label(y, weights))
        left_node = Node(value=self._common_label(y[left_idx], weights[left_idx]))
        right_node = Node(value=self._common_label(y[right_idx], weights[right_idx]))
        return Node(best_feat, best_value, left_node, right_node)

    def _common_label(self, y, weights):
        counts = Counter()
        for label, weight in zip(y, weights):
            counts[label] += weight
        return counts.most_common(1)[0][0]

    def _best_split(self, X, y, weights):
        best_gain = -np.inf
        split_idx, split_value = None, None
        for feat in range(self.n_features):
            if self.feature_types[feat] == "num":
                values = X[:, feat]
                median = np.median(values)
                left_idx = np.where(X[:, feat] <= median)[0]
                right_idx = np.where(X[:, feat] > median)[0]
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                gain = self._calc_gain(y, left_idx, right_idx, weights)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_value = median
            else:
                unique_values = np.unique(X[:, feat])
                for val in unique_values:
                    left_idx, right_idx = self._split(X, feat, val)
                    if len(left_idx) == 0 or len(right_idx) == 0:
                        continue
                    gain = self._calc_gain(y, left_idx, right_idx, weights)
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat
                        split_value = val
        return split_idx, split_value

    def _calc_gain(self, y, left_idx, right_idx, weights):
        left_y, right_y = y[left_idx], y[right_idx]
        left_w, right_w = weights[left_idx], weights[right_idx]
        if self.criterion == 'info_gain':
            return self._info_gain(y, left_y, right_y, weights, left_w, right_w)

    def _info_gain(self, y, left_y, right_y, parent_w, left_w, right_w):
        parent_entropy = self._entropy(y, parent_w)
        left_entropy = self._entropy(left_y, left_w)
        right_entropy = self._entropy(right_y, right_w)
        weighted_entropy = (
            np.sum(left_w) * left_entropy + np.sum(right_w) * right_entropy
        ) / np.sum(parent_w)
        return parent_entropy - weighted_entropy

    def _entropy(self, y, weights):
        total_weight = np.sum(weights)
        if total_weight == 0:
            return 0
        counts = Counter(y)
        entropy_value = 0.0
        for label in counts:
            p = (weights[y == label].sum()) / total_weight
            if p > 0:
                entropy_value -= p * np.log2(p)
        return entropy_value

    def _split(self, X, feat, value):
        if self.feature_types[feat] == "num":
            left_idx = np.where(X[:, feat] <= value)[0]
            right_idx = np.where(X[:, feat] > value)[0]
        else:
            left_idx = np.where(X[:, feat] == value)[0]
            right_idx = np.where(X[:, feat] != value)[0]
        return left_idx, right_idx

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if self.feature_types[node.feature] == "num":
            if x[node.feature] <= node.threshold:
                return self._traverse(x, node.left)
            else:
                return self._traverse(x, node.right)
        else:
            if x[node.feature] == node.threshold:
                return self._traverse(x, node.left)
            else:
                return self._traverse(x, node.right)

def adaboost_fit(X_train, y_train, X_test, y_test, T=500):
    n_samples, _ = X_train.shape
    weights = np.ones(n_samples) / n_samples
    stumps = []
    stump_weights = []
    train_errs = []
    test_errs = []
    stump_errs = []

    for t in range(T):
        stump = Stump()
        stump.fit(X_train, y_train, weights)
        y_pred = stump.predict(X_train)

        err = max(np.sum(weights[y_train != y_pred]), 1e-20)
        alpha = 0.5 * np.log((1 - err) / err)

        stumps.append(stump)
        stump_weights.append(alpha)

        weights *= np.exp(-alpha * y_train * y_pred)
        weights /= np.sum(weights)

        train_preds = adaboost_predict(X_train, stumps, stump_weights, t+1)
        test_preds = adaboost_predict(X_test, stumps, stump_weights, t+1)

        train_err = 1 - accuracy_score(y_train, train_preds)
        test_err = 1 - accuracy_score(y_test, test_preds)

        train_errs.append(train_err)
        test_errs.append(test_err)
        stump_errs.append(f"{(err / n_samples):.4f}")

    return train_errs, test_errs, stump_errs

def adaboost_predict(X, stumps, stump_weights, T=None):
    if T is None:
        T = len(stumps)

    stump_preds = np.array([stump.predict(X) for stump in stumps[:T]])
    weighted_sum = np.dot(stump_weights[:T], stump_preds)

    return np.where(weighted_sum > 0, 1, -1)

def binarize(X, feature_indices):
    for idx in feature_indices:
        median_value = np.median(X[:, idx])
        X[:, idx] = (X[:, idx] > median_value).astype(int)
    return X

def main():
    headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
               'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

    df_train = pd.read_csv("data/bank/train.csv", names=headers)
    df_test = pd.read_csv("data/bank/test.csv", names=headers)

    num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    X_train = df_train.drop('label', axis=1).values
    X_test = df_test.drop('label', axis=1).values
    X_train = binarize(X_train, [df_train.columns.get_loc(col) for col in num_cols])
    X_test = binarize(X_test, [df_test.columns.get_loc(col) for col in num_cols])

    y_train = df_train['label'].apply(lambda x: 1 if x == 'yes' else -1).values
    y_test = df_test['label'].apply(lambda x: 1 if x == 'yes' else -1).values

    T = 500
    train_errs, test_errs, stump_errs = adaboost_fit(X_train, y_train, X_test, y_test, T=T)

    print(f"Final training error: {train_errs[-1]:.4f}")
    print(f"Final test error: {test_errs[-1]:.4f}")

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, T + 1), train_errs, label="Train Err", linestyle='-')
    plt.plot(range(1, T + 1), test_errs, label="Test Err", linestyle='-')
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Train and Test Errors vs. T")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, T + 1), stump_errs, label="Stump Error", linestyle='-')
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.title("Stump Errors vs. T")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()