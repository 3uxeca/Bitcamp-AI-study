from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Lambda


#기온 데이터 읽어 들이기
kp = pd.read_csv('./data/test0822.csv', encoding="UTF-8")


# print(kp.shape) # 5479,9


#데이터를 학습 전용과 테스트 전용으로 분리하기
train_data = (kp[:3113])
test_data = (kp[3118:])
interval = 5
# print(train_data.shape) # 3113, 9
# print(test_data.shape) #2361, 9
# print(train_data.drop('date', axis = 1))


#과거 5일의 데이터를 기반으로 학습할 데이터 만들기
def make_data(data):
    x = []
    y = []
    temps = list(data)
    for i in range(len(temps)-(interval * 2)):
        xa = []
        ya = []
        for j in range(interval):
            xa.append(temps[i + j])
            ya.append(temps[interval + i + j])
        x.append(xa)
        y.append(ya)
    x = np.array(x)
    y = np.array(y)
    return(x, y)


train_data = np.array(train_data.drop('date', axis = 1))
test_data = np.array(test_data.drop('date', axis = 1))
train_x, train_y = make_data(train_data)
test_x, test_y = make_data(test_data)




print(train_x.shape) # 3103, 5, 8
print(test_x.shape) # 2351, 5, 8


train_x = train_x.reshape(-1, 5 * 8)
test_x = test_x.reshape(-1, 5 * 8)


print(train_x.shape) # 3103, 40
print(test_x.shape) # 2351, 40


from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
# scaler = MinMaxScaler()
scaler.fit(train_x)
train_x_scaled = scaler.transform(train_x)
test_x_scaled = scaler.transform(test_x)
print(test_x_scaled)


train_x = train_x.reshape(-1, 5, 8)
test_x = test_x.reshape(-1, 5,8)


print(train_x.shape) # 3103, 5, 8
print(test_x.shape) # 2351, 5, 8


model = Sequential()
model.add(LSTM(8, input_shape = (5, 8), activation = 'relu', return_sequences=True))
# model.add(LSTM(32, return_sequences=True, activation = 'relu'))
# model.add(LSTM(64, return_sequences=True, activation = 'relu'))
# model.add(LSTM(16, return_sequences=True, activation = 'relu'))
# model.add(LSTM(8, return_sequences=True, activation = 'relu'))
model.add(Lambda(lambda x: x[:, -5:, :]))


model.compile(optimizer = 'adam', loss = "mse", metrics=['acc', 'mse'])


history = model.fit(train_x, train_y, epochs=5, batch_size= 32, validation_data=(test_x, test_y))


pred_y = model.predict(test_x)


print(pred_y)


loss,_, _ = model.evaluate(test_x, test_y, batch_size = 10)


print(' mse = ', loss)


#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test[0], y_predict[0]))
print("RMSE : ", RMSE(test_y, pred_y))


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)


plt.plot(epochs, loss, 'bo', label = "Training loss")
plt.plot(epochs, val_loss, 'b', label = "Validation loss")
plt.title("Training and Validation loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.show()


pred_data = (kp[3108:3113])
print(pred_data)
pred_data = np.array(pred_data.drop('date', axis = 1))
print(pred_data)
pred_data = pred_data.reshape(-1, 5,8)
predict = model.predict(pred_data)
print(predict)


predict = np.array(predict, dtype = "int32")
print(predict.shape)
predict = predict.reshape(5, 8)
print(predict)
np.savetxt("test0822_swh.csv", predict, fmt='%d', delimiter = ",")

