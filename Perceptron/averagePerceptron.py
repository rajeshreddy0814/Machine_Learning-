import pandas as pd
import numpy as np

learning_rate = 0.05
max_epochs = 10

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

weights = np.zeros(X_train.shape[1])
cumulative_weights = np.zeros(X_train.shape[1])
total_updates = 0

for epoch in range(max_epochs):
    for i in range(len(X_train)):
        prediction = np.sign(np.dot(weights, X_train[i]))
        if prediction == 0:
            prediction = 1 

        if y_train[i] != prediction:
            weights += learning_rate * y_train[i] * X_train[i]
            cumulative_weights += weights
            total_updates += 1
        else:
            cumulative_weights += weights

average_weights = cumulative_weights / (len(X_train) * max_epochs)

def predict(X, weights):
    return np.sign(np.dot(X, weights))

predictions = predict(X_test, average_weights)
test_errors = np.sum(predictions != y_test)
average_error = test_errors / len(X_test)

print("Learned weight vector (average):", average_weights)
print("Average prediction error on test dataset:", average_error)
