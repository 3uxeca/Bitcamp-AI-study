import numpy as np
import pandas as pd
from ml import simple

train = pd.read_csv('./kaggle/cat/train.csv')
test = pd.read_csv('./kaggle/cat/test.csv')
# print(train.shape, test.shape) # (300000, 25), (200000, 24)

col = [c for c in train.columns if c not in ['id','target']]
for c in col:
    if train[c].nunique() > 2:
        enc1 = {e2:e1 for e1, e2 in enumerate(train[c].value_counts().index)} #counts
        enc2 = {e2:e1 for e1, e2 in enumerate(sorted(train[c].unique()))} #alpha
        enc3 = {e1:e2 for e1, e2 in train.groupby([c])['target'].agg('sum').rank(ascending=1).reset_index().values} #target_sum_rank
        enc4 = {e1:e2 for e1, e2 in train.groupby([c], as_index=False)['target'].agg('mean').values} #target_mean
        train[c+'enc1'] = train[c].map(enc1)
        train[c+'enc2'] = train[c].map(enc2)
        train[c+'enc3'] = train[c].map(enc3)
        train[c+'enc4'] = train[c].map(enc4)
        train.drop(columns=[c], inplace=True)
        test[c+'enc1'] = test[c].map(enc1)
        test[c+'enc2'] = test[c].map(enc2)
        test[c+'enc3'] = test[c].map(enc3)
        test[c+'enc4'] = test[c].map(enc4)
        test.drop(columns=[c], inplace=True)
    else:
        enc1 = {e2:e1 for e1, e2 in enumerate(sorted(train[c].unique()))} #alpha
        train[c] = train[c].map(enc1)
        test[c] = test[c].map(enc1) 

# print(train.shape, test.shape) # (300000, 79), (200000, 78)

data = simple.Data(train, test, 'id', 'target')
params = {'learning_rate': 0.005, 'max_depth': 7, 'boosting': 'gbdt', 
          'objective': 'binary', 'metric':'auc', 'seed': 4, 
          'num_iterations': 5000, 'early_stopping_round': 100, 
          'verbose_eval': 200, 'num_leaves': 64, 'feature_fraction': 0.9, 
          'bagging_fraction': 0.9, 'bagging_freq': 2}
sub = simple.Model(data, 'LGB', params, 0.2, 4).PRED
sub.to_csv('submission3.csv', index=False)