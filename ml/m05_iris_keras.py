import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC   # 분류모델
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris.csv", encoding='utf-8',
                        names=['a', 'b', 'c', 'd', 'y'])
# print(iris_data)
# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "y"] # .loc 레이블로 나누기(레이블은 실질적인 데이터가 아님)
x = iris_data.loc[:,["a", "b", "c", "d"]]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y) # 0 = setosa, 1 = versicolor, 2 = virginica
print(pd.value_counts(y_encoded))

# 범주형으로 전환
y_encoded = np_utils.to_categorical(y_encoded, 3)
print(y_encoded)
print(y_encoded.shape)

x_array = np.array(x)
print(x_array)
print(x_array.shape)
# print("==============================")
# print(x.shape) # (150, 4)
# print(y.shape) # (150,)
# print(x2.shape)
# print(y2.shape)

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x_array,y_encoded, test_size=0.2, train_size=0.8, shuffle=True)

# 모델링
model = Sequential()
model.add(Dense(32, input_dim=4, activation='relu'))
model.add(Dense(16)) # 여기서부턴 DNN연산모델
model.add(Dense(3, activation = 'softmax'))

model.summary()

# 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=100)

# 평가하기
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)

# decoding
y_predict = np.argmax(y_predict, axis=1).reshape(-1) # one-hot encoding한 값을 decoding하는 것 [0,0,1])=>[0,1,2]
y_predict = label_encoder.inverse_transform(y_predict) # int로 encoding했던 str을 다시 불러오는 것

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)
