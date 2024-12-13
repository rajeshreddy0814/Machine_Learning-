import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, classification_report


train_df = pd.read_csv('train_final.csv')
test_df = pd.read_csv('test_final.csv')


train_df.replace('?', np.nan, inplace=True)
test_df.replace('?', np.nan, inplace=True)


X = train_df.drop('income>50K', axis=1)
y = train_df['income>50K']


categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns


X[categorical_cols] = X[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))
test_df[categorical_cols] = test_df[categorical_cols].apply(lambda col: col.fillna(col.mode()[0]))


X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
test_encoded = pd.get_dummies(test_df.drop('ID', axis=1), columns=categorical_cols, drop_first=True)


X_encoded, test_encoded = X_encoded.align(test_encoded, join='left', axis=1, fill_value=0)


scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
test_encoded[numerical_cols] = scaler.transform(test_encoded[numerical_cols])


X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)


ada_model.fit(X_train, y_train)


y_val_pred = ada_model.predict_proba(X_val)[:, 1]


auc_score = roc_auc_score(y_val, y_val_pred)
print(f'Validation AUC Score: {auc_score:.4f}')


y_val_class = ada_model.predict(X_val)
print(classification_report(y_val, y_val_class))


test_predictions = ada_model.predict_proba(test_encoded)[:, 1]


submission = pd.DataFrame({
    'ID': test_df['ID'],
    'Prediction': test_predictions
})


submission.to_csv('submission.csv', index=False)

print("Submission file 'adaboost_submission.csv' created successfully.")
