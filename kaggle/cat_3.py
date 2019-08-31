# Importing required modules
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import roc_auc_score
import numpy as np

# 데이터 불러오기
data_train=pd.read_csv('./kaggle/cat/train.csv')
data_test=pd.read_csv('./kaggle/cat/test.csv')

# 데이터
for col in data_train.columns:
    print(col," --- ",len(data_train[col].value_counts()),"--- ",data_train[col].dtype)

print(data_train['target'].value_counts())

y = data_train['target']
data_id = data_test['id']

data_train=data_train.drop(['id','target'],axis=1)
data_test=data_test.drop(['id'],axis=1)

cate_cols = [cols for cols in data_train.columns if data_train[cols].dtype == 'object']

# Label Encoding the categorical columns 
encoder = LabelEncoder()
for col in cate_cols:
    data_train[col] = pd.DataFrame(encoder.fit_transform(data_train[col]))
    data_test[col] = pd.DataFrame(encoder.fit_transform(data_test[col]))   
x_train,x_test,y_train,y_test = train_test_split(data_train,y,random_state=1)

# 학습
clf = XGBClassifier(n_estimators=200,scale_pos_weight=2,random_state=1,colsample_bytree=0.5)
clf.fit(x_train,y_train)

predictions = clf.predict_proba(x_test)[:,1]

# 평가
score = roc_auc_score(y_test,predictions)
print(score)

predict = clf.predict_proba(data_test)[:,1]

submission = pd.DataFrame({'id': data_id, 'target': predict})
submission.to_csv('submission2.csv', index=False)