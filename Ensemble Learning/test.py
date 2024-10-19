import pandas as pd
import DT as dt
import numpy as np
import matplotlib.pyplot as plt
import os

# Load train data
columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
types = {'age': int, 
        'job': str, 
        'marital': str, 
        'education': str,
        'default': str,
        'balance': int,
        'housing': str,
        'loan': str,
        'contact': str,
        'day': int,
        'month': str,
        'duration': int,
        'campaign': int,
        'pdays': int,
        'previous': int,
        'poutcome': str,
        'y': str}

# File paths for train and test data
train_file_path = os.path.abspath('data/bank/train.csv')
test_file_path = os.path.abspath('data/bank/test.csv')

# Load datasets
train_df = pd.read_csv(train_file_path, names=columns, dtype=types)
train_size = len(train_df.index)

# Process data: convert numeric to binary
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for col in numeric_cols:
    median_value = train_df[col].median()
    train_df[col] = train_df[col].apply(lambda x: 0 if x < median_value else 1)

# Load test data
test_df = pd.read_csv(test_file_path, names=columns, dtype=types)
test_size = len(test_df.index)

for col in numeric_cols:
    median_value = test_df[col].median()
    test_df[col] = test_df[col].apply(lambda x: 0 if x < median_value else 1)

# Set features and labels
features_dict = {'age': [0, 1],  # converted to binary
        'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'], 
        'marital': ['married','divorced','single'], 
        'education': ['unknown', 'secondary', 'primary', 'tertiary'],
        'default': ['yes', 'no'],
        'balance': [0, 1],  # converted to binary
        'housing': ['yes', 'no'],
        'loan': ['yes', 'no'],
        'contact': ['unknown', 'telephone', 'cellular'],
        'day': [0, 1],  # converted to binary
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'duration': [0, 1],  # converted to binary
        'campaign': [0, 1],  # converted to binary
        'pdays': [0, 1],  # converted to binary
        'previous': [0, 1],  # converted to binary
        'poutcome': ['unknown', 'other', 'failure', 'success']}
label_dict = {'y': ['yes', 'no']}

# Number of iterations for Random Forest
num_iterations = 1000

train_error_rates = [0 for x in range(num_iterations)]
test_error_rates = [0 for x in range(num_iterations)]
train_predictions_sum = np.array([0 for x in range(train_size)])
test_predictions_sum = np.array([0 for x in range(test_size)])

for iteration in range(num_iterations):
    # Sample with replacement
    sampled_data = train_df.sample(frac=0.5, replace=True, random_state=iteration)
    
    # ID3 Decision Tree
    dt_generator = dt.ID3(feature_selection=0, max_depth=17, subset=2)
    
    # Generate decision tree
    decision_tree = dt_generator.generate_decision_tree(sampled_data, features_dict, label_dict)

    ## Predictions
    # Train predictions
    predicted_labels_train = dt_generator.classify(decision_tree, train_df)
    predicted_labels_train = np.array(predicted_labels_train.tolist())
    predicted_labels_train[predicted_labels_train == 'yes'] = 1
    predicted_labels_train[predicted_labels_train == 'no'] = -1
    predicted_labels_train = predicted_labels_train.astype(int)
    train_predictions_sum += predicted_labels_train
    predicted_labels_train = predicted_labels_train.astype(str)
    predicted_labels_train[train_predictions_sum > 0] = 'yes'
    predicted_labels_train[train_predictions_sum <= 0] = 'no'
    train_df['predicted_y'] = pd.Series(predicted_labels_train)

    # Train accuracy and error
    train_accuracy = train_df.apply(lambda row: 1 if row['y'] == row['predicted_y'] else 0, axis=1).sum() / train_size
    train_error = 1 - train_accuracy
    train_error_rates[iteration] = train_error

    # Test predictions
    predicted_labels_test = dt_generator.classify(decision_tree, test_df)
    predicted_labels_test = np.array(predicted_labels_test.tolist())
    predicted_labels_test[predicted_labels_test == 'yes'] = 1
    predicted_labels_test[predicted_labels_test == 'no'] = -1
    predicted_labels_test = predicted_labels_test.astype(int)
    test_predictions_sum += predicted_labels_test
    predicted_labels_test = predicted_labels_test.astype(str)
    predicted_labels_test[test_predictions_sum > 0] = 'yes'
    predicted_labels_test[test_predictions_sum <= 0] = 'no'
    test_df['predicted_y'] = pd.Series(predicted_labels_test)

    # Test accuracy and error
    test_accuracy = test_df.apply(lambda row: 1 if row['y'] == row['predicted_y'] else 0, axis=1).sum() / test_size
    test_error = 1 - test_accuracy
    test_error_rates[iteration] = test_error

    print(f'Iteration: {iteration}, Train Error: {train_error_rates[iteration]}, Test Error: {test_error_rates[iteration]}')

# Plot error rates
plt.figure()
plt.title('Random Forest, Feature Subset = 2')
plt.xlabel('Iteration')
plt.ylabel('Error Rate')
plt.plot(train_error_rates, 'b')
plt.plot(test_error_rates, 'r')
plt.legend(['Train Error', 'Test Error'])
plt.savefig('random_forest_errors.png')
