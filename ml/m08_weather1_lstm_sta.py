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

batch_size = 2

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

train_x = train_x.reshape(train_x.shape[0], 6, 1)
test_x = test_x.reshape(test_x.shape[0], 6, 1)


# 정규화
# scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler.fit(train_x)
# train_x_scaled = scaler.transform(train_x)
# test_x_scaled = scaler.transform(test_x)
# print("train_x_scaled:",train_x_scaled[:10])
# print("test_x_scaled:",test_x_scaled[:10])

print(train_x.shape) # (3646, 6, 1)
print(test_x.shape) # (360, 6, 1)
print(train_y.shape) # (3646,)
print(test_y.shape) # (360,)


# 모델의 설정
model = Sequential()
model.add(LSTM(10, batch_input_shape=(batch_size,6,1), stateful=True))

model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# 모델 최적화 설정
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')

# 학습
model.compile(optimizer='adam', loss='mse', metrics = ['mse'])

num_epochs = 5
history_l = []
for epoch_idx in range(num_epochs):
    print('epochs : ' + str(epoch_idx))
    history = model.fit(train_x, train_y, epochs=100, batch_size=batch_size,
                        verbose=2, shuffle=False, # 데이터를 섞지않고(초기화하지않고) 유지하겠다. epoch하나 끝나면 훈련상태를 그대로 다시 가져오겠다.
                        validation_data=(test_x, test_y),
                        callbacks=[early_stopping])
    model.reset_states() # 리셋했다고해서 지워지는건 아니다. 상태유지lstm에서는 꼭 넣어줘야함
    history_l.append(history)

print(history)


#4. 평가 예측
mse, _ = model.evaluate(train_x, train_y, batch_size=batch_size)
print("mse : ", mse)
model.reset_states()

y_predict = model.predict(test_x, batch_size=batch_size)

print(y_predict[0:5])

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(test_y, y_predict): # test_y, y_predict의 차이를 비교하기 위한 함수
    return np.sqrt(mean_squared_error(test_y, y_predict)) # np.sqrt 제곱근씌우기
print("RMSE : ", RMSE(test_y, y_predict))

# R2(결정계수) 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(test_y, y_predict)
print("R2 : ", r2_y_predict)

# mse에 대한 히스토리 요약
for i in range(len(history_l)):
   plt.plot(history_l[i].history['mean_squared_error'])
plt.title('model mse')
plt.ylabel('mse')
plt.xlabel('epochs')
plt.legend(['train', 'test'], loc='upper right') # for문 돌때마다 발생하는 값들을 리스트로 넣는방법 찾기.
plt.show()
