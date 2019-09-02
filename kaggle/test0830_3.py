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
from xgboost import XGBClassifier

# 데이터
X = np.load("cat_X.npy")
y = np.load("cat_y.npy")
test_data = np.load("test_data.npy")
test_id = np.load("test_id.npy")
sample_submission = pd.read_csv('./kaggle/cat/sample_submission.csv', index_col='id')
# print(X.shape) # (300000, 23)
# print(y.shape) # (300000, )
# print(X[:10])

# Scale 
# X = StandardScaler().fit_transform(X)
# print(X[:10])

# 학습 & 평가 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42) 

# print(X_train.shape, X_test.shape) # (240000, 23) / (60000, 23)
# print(y_train.shape, y_test.shape) # (240000, ) / (60000, )


# 학습
clf = XGBClassifier(n_estimators=900, scale_pos_weight=2, random_state=1024, 
                    colsample_bytree=0.5, objective="binary:logistic", learning_rate=0.05,
                    early_stopping_rounds=200, reg_alpha=0.3, reg_lambda=0.8, subsample=0.7, n_jobs=-1)
clf.fit(X_train, y_train)

# clf = GradientBoostingClassifier(n_estimators=200,random_state=1)
# clf.fit(X_train, y_train)

predictions = clf.predict_proba(X_test)[:,1]

# 평가
score = roc_auc_score(y_test,predictions)
print(score)

predict = clf.predict_proba(test_data)[:,1]
print(predict)

submission = pd.DataFrame({'id': test_id, 'target': predict})
submission.to_csv('submission01.csv', index=False)