import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('./kaggle/cat/train.csv')
test_data = pd.read_csv('./kaggle/cat/test.csv')
submission = pd.read_csv('./kaggle/cat/sample_submission.csv', index_col='id')


# print(train_data.shape) # (300000, 25)
# print(test_data.shape) # (200000, 24)
# print(train_data.columns)
# print(test_data.columns)
# print(train_data.head())
# print(test_data.head())

target_ = train_data['target']

from sklearn.preprocessing import LabelEncoder
len_uniques = []
for c in train_data.columns.drop(['target','id']):
    le = LabelEncoder()
    le.fit(pd.concat([train_data[c], test_data[c]])) 
    train_data[c] = le.transform(train_data[c])
    test_data[c] = le.transform(test_data[c])
    len_uniques.append(len(le.classes_))

# print("train data.shape: {}  test data.shape: {}".format(train_data.shape, test_data.shape))
# train data.shape: (300000, 25)  test data.shape: (200000, 24)

X = train_data.drop(['target', 'id'], axis=1)
y = train_data['target']
test_data1 = test_data.drop(['id'], axis=1)
test_id = test_data['id']
print(X) # (300000, 23)
print(y) # (300000, )
print(test_data1)


# np.save("cat_X.npy", X)
# np.save("cat_y.npy", y)
# np.save("test_data.npy", test_data1)
# np.save("test_id.npy", test_id)

# X_load = np.load("cat_X.npy")
# print(X_load.shape)

'''
# print("X.shape: {}  y.shape: {}".format(X.shape, y.shape))
# X.shape: (300000, 23)  y.shape: (300000,)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.20, random_state=42) 

logreg = LogisticRegression() 
logreg.fit(X_train, y_train)
log_pre=logreg.predict(X_test)
# print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))
# print('Accuracy : ',accuracy_score(y_test,log_pre))
# Test score: 0.72
# Accuracy :  0.7199833333333333

rfcl = RandomForestClassifier(n_estimators=70, n_jobs=-1, min_samples_leaf=5)
rfcl.fit(X_train, y_train)
rfcl_pre=rfcl.predict(X_test)
# print("Test score: {:.2f}".format(rfcl.score(X_test, y_test)))
# print('Accuracy : ',accuracy_score(y_test,rfcl_pre))
# Test score: 0.73
# Accuracy :  0.7328

gbcl = GradientBoostingClassifier()
gbcl.fit(X_train, y_train)
gbcl_pre=gbcl.predict(X_test)
print("Test score: {:.2f}".format(gbcl.score(X_test, y_test)))
print('Accuracy : ',accuracy_score(y_test,gbcl_pre))
# Test score: 0.74
# Accuracy :  0.7377

from sklearn.model_selection import StratifiedKFold 
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_score(gbcl, X_train, y_train, cv=skfolds, scoring="accuracy") 
test_d = test_data.set_index('id')

submission['target'] = gbcl.predict_proba(test_d)[:, 1]
# print(submission.head(10))

submission.to_csv('submission_sjh.csv')
'''