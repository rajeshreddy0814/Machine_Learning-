import pandas as pd
import numpy as np

headers = [
    "Variance_Wavelet", 
    "Skewness_Wavelet", 
    "Curtosis_Wavelet", 
    "Entropy", 
    "Label"
]

train_data = pd.read_csv("bank-note/train.csv", header=None, names=headers)
test_data = pd.read_csv("bank-note/test.csv", header=None, names=headers)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

learning_rate = 0.05
max_epochs = 10
n_samples, n_features = X_train.shape

weights = np.zeros(n_features)
weight_vectors = []
counts = []
epoch_misclassifications = []

for epoch in range(max_epochs):
    mistakes = 0

    for i in range(n_samples):
        prediction = np.sign(np.dot(weights, X_train[i]))
        if prediction == 0:
            prediction = -1

        if y_train[i] != prediction:
            weights += learning_rate * y_train[i] * X_train[i]
            mistakes += 1

    weight_vectors.append(weights.copy())
    counts.append(n_samples - mistakes)
    epoch_misclassifications.append(mistakes)

predictions = []
for x in X_test:
    vote_sum = 0
    for w, count in zip(weight_vectors, counts):
        vote = np.sign(np.dot(w, x))
        vote_sum += count * vote
    predictions.append(np.sign(vote_sum))

test_errors = np.sum(np.array(predictions) != y_test)
average_error = test_errors / len(y_test)

print("\nWeight vectors and their correct counts:")
for i, (w, count) in enumerate(zip(weight_vectors, counts)):
    print(f"Weight Vector {i + 1}: {w}, Correct Count: {count}")

print(f"\nAverage prediction error on test dataset: {average_error:.2f}")
