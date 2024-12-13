import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

# Load the datasets
train_df = pd.read_csv('files/train_final.csv')
test_df = pd.read_csv('files/test_final.csv')

# Replace missing values marked as '?' with NaN for easier handling
train_df.replace('?', np.nan, inplace=True)
test_df.replace('?', np.nan, inplace=True)

# Separate the target variable in the training dataset
X = train_df.drop('income>50K', axis=1)
y = train_df['income>50K']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Fill missing values for categorical columns with the most frequent value
X[categorical_cols] = X[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))
test_df[categorical_cols] = test_df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))

# Encode categorical variables using OneHotEncoder
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test_df.drop('ID', axis=1), columns=categorical_cols, drop_first=True)

# Align train and test datasets to have the same columns
X_encoded, test_encoded = X_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

# Standardize numerical features
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
test_encoded[numerical_cols] = scaler.transform(test_encoded[numerical_cols])

# Split the training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Predict probabilities on the validation set
y_val_pred = rf_model.predict_proba(X_val)[:, 1]

# Calculate the AUC score
auc_score = roc_auc_score(y_val, y_val_pred)
print(f'Validation AUC Score: {auc_score:.4f}')

# Print classification report for additional evaluation metrics
y_val_class = rf_model.predict(X_val)
print(classification_report(y_val, y_val_class))

# Make predictions on the test dataset
test_predictions = rf_model.predict_proba(test_encoded)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Prediction': test_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)

print("Submission file 'random_forest_submission.csv' created successfully.")
