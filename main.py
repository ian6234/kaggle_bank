import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 1 - load train/test data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
# 2 - Encoding data - dummy variables/one-hot encoding etc.

train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

print(train_df.head())
X = train_df.drop(['y'], axis=1)
y = train_df['y']

# 3 - model selection - logistic regression suitable for probability forecasting
model = LogisticRegression(random_state=0, max_iter=300).fit(X, y)

probabilities = model.predict_proba(test_df)
print(probabilities)
# 4 - save output to csv for submission
output = {'id': list(test_df['id']), 'y': [p[1] for p in probabilities]}
submission_df = pd.DataFrame(output)

submission_df.to_csv('submission.csv', index=False)
