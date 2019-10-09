import numpy as np 
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 데이터
submission = pd.read_csv('sample_submission.csv', index_col='id')
X = np.load("cat_X.npy")
y = np.load("cat_y.npy")
# print(X.shape) # (300000, 23)
# print(y.shape) # (300000, )
# print(X[:10])

# Scale 
X = StandardScaler().fit_transform(X)
# print(X[:10])

# 학습 & 평가 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.20, random_state=42) 

# print(X_train.shape, X_test.shape) # (240000, 23) / (60000, 23)
# print(y_train.shape, y_test.shape) # (240000, ) / (60000, )

'''
# 모델링
model = Sequential()
model.add(Dense(32, input_dim=23, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()
'''

# ROC 정의
from sklearn import metrics
from keras import backend as K
import tensorflow as tf
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# 학습
from xgboost import XGBClassifier
xgbcl = XGBClassifier()
xgbcl.fit(X_train, y_train)
xgbcl_pre = xgbcl.predict(X_test)
# gbcl = GradientBoostingClassifier()
# gbcl.fit(X_train, y_train)
# gbcl_pre=gbcl.predict(X_test)
print("Test score : {:.2f}".format(xgbcl.score(X_test, y_test)))
print('Accuracy : ',accuracy_score(y_test,xgbcl_pre))
print('roc_auc_score : ',roc_auc_score(y_test, xgbcl_pre))
# Test score: 0.74
# Accuracy :  0.7377

from sklearn.model_selection import StratifiedKFold 

skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_score(xgbcl, X_train, y_train, cv=skfolds, scoring="accuracy") 
# test_d = test_data.set_index('id')


submission['target'] = pd.DataFrame(xgbcl_pre[:, 1])
# print(submission.head(10))

submission.to_csv('submission_sjh2.csv')