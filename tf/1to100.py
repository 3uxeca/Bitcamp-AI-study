# 1 ~ 100 까지의 숫자를 이용해서
# 6개씩 잘라서 rnn 구성
# train, test 분리할 것

# 1,2,3,4,5,6 : 7
# 2,3,4,5,6,7 : 8
# 3,4,5,6,7,8 : 9
# ...
# 94,95,96,97,98,99 : 100

# predict : 101 ~ 110 까지 예측하시오.
# 지표 : RMSE

import numpy as np 

x_data = np.arange(1,101)
x_test = np.arange(101,110)
y_test = np.array([107, 108, 109, 110])
size = 6
pre_day = 1

# train / label data split



from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization

model = Sequential() 
model.add(LSTM(50, input_shape=(6,1), activation='relu'))
                
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# early_stopping = EarlyStopping(monitor='loss', patience=50, mode='min')
# tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0,
                    #   write_graph=True, write_images=True)

model.fit(x_train, y_train, epochs=50, batch_size = batch_size)

mse, acc = model.evaluate(x_train, y_train, batch_size=1)
print("mse : ", mse)
print("acc : ", acc)

y_predict = model.predict(x_test, batch_size=batch_size)

print(y_predict[0:11])

x_input = np.array([95,96,97,98,99,100])
x_input = x_input.reshape(1, len(x_input), 1)

predict = model.predict(x_input)

print(predict[:11])

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # y_test, y_predict의 차이를 비교하기 위한 함수
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt 제곱근씌우기
print("RMSE : ", RMSE(y_test, y_predict))

