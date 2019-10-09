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
X = np.load("cat_X.npy")
y = np.load("cat_y.npy")
test = np.load("cat_test.npy")
test_id = np.load("test_id.npy")
sample_submission = pd.read_csv('./kaggle/cat/sample_submission.csv', index_col='id')
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

# 모델링
model = Sequential()
model.add(Dense(32, input_dim=23, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# ROC 정의
from sklearn import metrics
from keras import backend as K
import tensorflow as tf
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy', auc])
model.fit(X_train, y_train, epochs=1, batch_size=100)

# 평가
loss, acc, auc = model.evaluate(X_test, y_test, batch_size=1)
prediction = model.predict(X_test)
score = roc_auc_score(y_test, prediction)

print('loss : ', loss)
print('acc : ', acc)
print('auc : ', auc)
print('score : ', score)

predict = model.predict(test)[:,]
print('predict(test) : \n', predict)

# csv 저장
submission = pd.DataFrame({'id': test_id, 'target': predict})
submission.to_csv('submission_sjh2.csv', index=False)
# sample_submission['target'] = pd.DataFrame(predict).to_csv("submission_sjh2.csv")