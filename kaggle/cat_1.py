import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.metrics import roc_auc_score

train=pd.read_csv('./kaggle/cat/train.csv')
test=pd.read_csv('./kaggle/cat/test.csv')
sample_submission = pd.read_csv('./kaggle/cat/sample_submission.csv')

target_var = train['target']
train.drop(['id', 'target'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)

encoder = LabelEncoder()
categorical_features = train.columns.tolist()
for each in categorical_features:
    train[each] = encoder.fit_transform(train[each])
    
test_cat_features = test.columns.tolist()
for col in test_cat_features:
    test[col] = encoder.fit_transform(test[col])

# feature scaling
scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)

# print(train[:10])
# print('----------------')
# print(test[:10])

# np.save("train.npy", train)
# np.save("test.npy", test)
# print(train.shape)
xgb_clf = xgboost.XGBClassifier(n_estimators=900, 
                            n_jobs=-1, 
                            subsample=0.7,
                            max_depth=8,
                            reg_alpha=0.3, 
                            reg_lambda=0.8, 
                            random_state=1024, 
                            learning_rate=0.05,
                            metric = 'auc',
                        #     tree_method= 'gpu_hist', 
                            objective="binary:logistic",
                            verbose=500,
                            early_stopping_rounds=200)

X = train
y = target_var
skf = StratifiedKFold(n_splits=5, random_state=1024, shuffle=False)

for train_index, val_index in skf.split(X, y):
  X_train, X_val = X[train_index], X[val_index]
  y_train, y_val = y[train_index], y[val_index]
  xgb_clf.fit(X_train, y_train)

predictions = xgb_clf.predict_proba(X_val)[:,1]
score = roc_auc_score(y_val,predictions)
print(score)