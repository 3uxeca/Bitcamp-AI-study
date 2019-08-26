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

a = np.array(range(1,101))

batch_size = 75
size = 7
def split_7(seq, size): # array의 데이터를 5개씩 잘라서 [1,2,3,4,5,6,7,8]
    aaa = []
    for i in range(len(a)-size + 1): # range(6) = 0~5 ==>>> 자른 갯수 + 1 = 행의 갯수
        subset = a[i:(i+size)]
        aaa.append(subset)
        #aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa) 

dataset = split_7(a, size)
print("====================")
# print(dataset)

x_train = dataset[:,0:6]
y_train = dataset[:,6]
# print(x_train[:5])
# print(y_train[:5])

x_train = np.reshape(x_train, (len(x_train), size-1, 1))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.2, shuffle=False)

# print(x_train.shape) # (75, 6, 1)
# print(y_train.shape) # (75, )
# print(y_train)
# print(x_test.shape) # (19, 6, 1)
# print(y_test.shape) # (19, )
# print(x_test[:10])
# print(y_test)


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

