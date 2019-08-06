import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# 데이터 읽어 들이기
wine = pd.read_csv("./data/winequality-white.csv", sep=";", encoding="utf-8")

# 데이터를 레이블과 데이터로 분리하기
y = wine["quality"]
x = wine.drop("quality", axis=1) # "quality"라는 column만 drop하고 나머지는 x값이 된다.

x = np.array(x)
y = np.array(y)

print(y[:20])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# print(y_encoded)
# print(y_encoded.shape) #(4898,7)

# 범주형으로 전환
y_encoded = np_utils.to_categorical(y_encoded, 7)
# print(y_encoded)
# print(y_encoded.shape)
# print(x[:10])

# 정규화
# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
# print(x_scaled[:10])

x_train, x_test, y_train, y_test = train_test_split(
                                    x_scaled, y_encoded, test_size=0.2)

# print(x_train.shape) # (3918, 11) 
# print(x_test.shape) # (980, 11)
# print(y_train.shape) # (3918,7)
# print(y_test.shape) # (980,7)


# 모델의 설정
model = Sequential()
model.add(Dense(50, input_dim=11, activation='relu'))
model.add(Dense(100))
model.add(Dense(60))
model.add(Dense(60))
model.add(Dense(7, activation='softmax'))

model.summary()

# 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=100)

# 평가하기
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis=1).reshape(-1)
y_predict = label_encoder.inverse_transform(y_predict) # int로 encoding했던 str을 다시 불러오는 것

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict[:100])

