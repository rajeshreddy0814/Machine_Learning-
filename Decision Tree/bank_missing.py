import pandas as pd
import numpy as np
from collections import Counter

column_headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']


df_train = pd.read_csv("datasets/bank/train.csv", names=column_headers)
df_test = pd.read_csv("datasets/bank/test.csv", names=column_headers)

def fill_unknowns(df):
    for column in df.columns:
        most_common = df[column].mode()[0]
        df[column] = df[column].replace("unknown", most_common)
    return df

df_train = fill_unknowns(df_train)
df_test = fill_unknowns(df_test)

X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values
X_test = df_test.drop('label', axis=1).values
y_test = df_test['label'].values

class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, is_numerical=False):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.prediction = None
        self.is_numerical = is_numerical

class DecisionTree:
    def __init__(self, max_depth=None, criterion='information_gain'):
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
            if np.issubdtype(values.dtype, np.number):  # Numerical feature
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
            else:  # Categorical feature
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
        elif self.criterion == 'majority_error':
            return self._majority_error(y, left_y, right_y)
        elif self.criterion == 'gini_index':
            return self._gini_index_gain(y, left_y, right_y)

    def _information_gain(self, y, left_y, right_y):
        parent_entropy = self._entropy(y)
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        child_entropy = (n_left / n) * self._entropy(left_y) + (n_right / n) * self._entropy(right_y)
        return parent_entropy - child_entropy

    def _majority_error(self, y, left_y, right_y):
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        majority_left = self._most_common_label(left_y)
        majority_right = self._most_common_label(right_y)

        majority_error = (
            (n_left / n) * (1 - (np.sum(left_y == majority_left) / n_left)) +
            (n_right / n) * (1 - (np.sum(right_y == majority_right) / n_right))
        )

        return 1 - majority_error

    def _gini_index_gain(self, y, left_y, right_y):
        gini_parent = self._gini_index(y)
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)

        weighted_gini = (n_left / n) * self._gini_index(left_y) + (n_right / n) * self._gini_index(right_y)
        return gini_parent - weighted_gini

    def _gini_index(self, y):
        counter = Counter(y)
        probabilities = np.array(list(counter.values())) / len(y)
        return 1 - np.sum(probabilities ** 2)

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


# Function to evaluate accuracy at each depth for different criteria
def evaluate_tree(depths, X_train, y_train, X_test, y_test):
    results = []

    for depth in depths:
        row = [depth] 
        for criterion in ['information_gain', 'majority_error', 'gini_index']:
            
            tree = DecisionTree(max_depth=depth, criterion=criterion)
            tree.fit(X_train, y_train)

            # Calculate accuracies
            train_accuracy = np.mean(tree.predict(X_train) == y_train)
            test_accuracy = np.mean(tree.predict(X_test) == y_test)

           
            row.append(round(1 - train_accuracy, 3))  # Majority error for train
            row.append(round(1 - test_accuracy, 3))   # Majority error for test

        results.append(row)

    return results


if __name__ == "__main__":
    depths = range(1, 17) 
    results = evaluate_tree(depths, X_train, y_train, X_test, y_test)


    headers = [
        "Depth", 
        "Info Gain (Train)", "Info Gain (Test)", 
        "Majority Error (Train)", "Majority Error (Test)", 
        "Gini Index (Train)", "Gini Index (Test)"
    ]

    column_widths = [max(len(header), max(len("{:.3f}".format(row[i])) for row in results)) for i, header in enumerate(headers)]

    header_row = " | ".join(f"{header:<{column_widths[i]}}" for i, header in enumerate(headers))
    print(header_row)
    print("-" * (len(header_row) + 2))
    for row in results:
        formatted_row = ["{:.3f}".format(x) for x in row]  # Format values to three decimal places
        print(" | ".join(f"{value:<{column_widths[i]}}" for i, value in enumerate(formatted_row)))
