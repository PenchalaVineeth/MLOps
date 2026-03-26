import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login

api = HfApi(token=os.getenv('HF_TOKEN'))
dataset_path = 'hf://datasets/vineeth32/Bank-Customer-Churn/bank_customer_churn.csv'
bank_dataset = pd.read_csv(dataset_path)
print('Dataset Loaded Successfully.')

target = 'Exited'

numeric_features = [
    'CreditScore',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary'
]

categorical_features = [
    'Geography',
]

X = bank_dataset[numeric_features+categorical_features]
y = bank_dataset[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

files = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']

for file_path in files:
  api.upload_file(
      path_or_fileobj=file_path,
      path_in_repo=file_path.split('/')[-1],
      repo_id='vineeth32/Bank-Customer-Churn',
      repo_type='dataset',
  )
