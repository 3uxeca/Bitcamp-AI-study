from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# 기온 데이터 읽어 들이기
df = pd.read_csv('./data/tem10y.csv', encoding="utf-8")

# 데이터를 학습 전용과 테스트 전용으로 분리하기
train_year = (df["연"] <= 2015)
test_year = (df["연"] >= 2016)
interval = 6

# 6개씩 잘라서 사용한 시계열 => LSTM으로 바꿀 수 있다.

# 과거 6일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = [] # 학습 데이터
    y = [] # 결과
    temps = list(data["기온"])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x, y)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

train_x = np.array(train_x)
test_x = np.array(test_x)
train_y = np.array(train_y)
test_y = np.array(test_y)

# print(train_x[:10])
# print(train_y[:10])
# label_encoder = LabelEncoder()
# train_y = label_encoder.fit_transform(train_y)
# test_y = label_encoder.fit_transform(test_y)
# print(train_y[:10])
# print(test_y[:10])

# 정규화
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler.fit(train_x)
# train_x_scaled = scaler.transform(train_x)
# test_x_scaled = scaler.transform(test_x)
# print("train_x_scaled:",train_x_scaled[:10])
# print("test_x_scaled:",test_x_scaled[:10])

# print(train_x_scaled.shape) # (3646, 6)
# print(test_x_scaled.shape) # (360, 6)
# print(train_y.shape) # (3646,)
# print(test_y.shape) # (360,)


# 모델의 설정
model = Sequential()
model.add(Dense(64, input_dim=6, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# 모델 최적화 설정
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')

# 학습
model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
model.fit(train_x, train_y, epochs=100, batch_size=20, 
          callbacks=[early_stopping] )

# 평가하기
loss, acc = model.evaluate(test_x, test_y, batch_size=1)
y_predict = model.predict(test_x)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(test_y, y_predict): # test_y, y_predict의 차이를 비교하기 위한 함수
    return np.sqrt(mean_squared_error(test_y, y_predict)) # np.sqrt 제곱근씌우기
print("RMSE : ", RMSE(test_y, y_predict))

# R2(결정계수) 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(test_y, y_predict)
print("R2 : ", r2_y_predict)
