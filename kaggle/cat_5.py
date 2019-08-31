import os
import gc
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from pprint import pprint
from tqdm import tqdm

train = pd.read_csv('./kaggle/cat/train.csv', index_col='id')
test = pd.read_csv('./kaggle/cat/test.csv', index_col='id')
sample_submission = pd.read_csv('./kaggle/cat/sample_submission.csv', index_col='id')

y_train = train.pop('target')

# Simple label encoding
for c in tqdm(train.columns):
    le = LabelEncoder()
    # this is cheating in real life; you won't have the test data ahead of time ;-)
    le.fit(pd.concat([train[c], test[c]])) 
    train[c] = le.transform(train[c])
    test[c] = le.transform(test[c])

X_train, X_val, y_train, y_val = train_test_split(
    train, y_train, test_size=0.2, random_state=2019)

clf = xgb.XGBClassifier(
    learning_rate=0.05,
    n_estimators=50000, # Very large number
    seed=2019,
    reg_alpha=5,
    eval_metric='auc',
    # tree_method='gpu_hist'
)
clf.fit(
    X_train, 
    y_train, 
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=50,
    verbose=50
)    

sample_submission['target'] = clf.predict_proba(test, ntree_limit=clf.best_ntree_limit)[:, 1]
sample_submission.to_csv('submission4.csv')