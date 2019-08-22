import numpy as np
import pandas as pd

# Data
df = pd.read_csv("./data/test0822.csv", sep=",")
# print(df[:10])
# print(df.shape) # (5479, 9)

# print(df['kp_0h'].value_counts())

# Train, test data split
df = df.drop("date", 1)
df_train = df.loc[0:3112]
df_test = df.loc[3118:]

# print(df[3108:3113])
# print(df_train) # 1999-01-01 ~ 2007-07-10 기간의 데이터. "date" 열 삭제.
# print(df_test) # 2007-07-16 ~ 2013-12-31 기간의 데이터. "date" 열 삭제.
# print(df_train.shape) # (3113, 8) 
# print(df_test.shape) # (2361, 8)

interval = 5
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

train_data = np.array(df_train)
test_data = np.array(df_test)
x_train, y_train = make_data(train_data)
x_test, y_test = make_data(test_data)

# print(x_train.shape) # (3103, 5, 8)
# print(x_test.shape) # (2351, 5, 8)
# print(y_train.shape) # (3103, 5, 8)
# print(y_test.shape) # (2351, 5, 8)

# Data scale
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
# scale = MinMaxScaler()
# scale.fit(x_train)
# x_train = scale.transform(x_train)
# x_test = scale.transform(x_test)

# print(x_train)
# print(x_test)


# Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, BatchNormalization
from keras import regularizers
model = Sequential()

model.add(LSTM(7, input_shape=(5, 8), activation='relu', return_sequences=True))
# model.add(LSTM(40, activation='relu',  kernel_regularizer = regularizers.l2(0.01), return_sequences=True))
# model.add(LSTM(40, activation='relu', kernel_regularizer = regularizers.l2(0.01), return_sequences=True))
# model.add(BatchNormalization())
# model.add(LSTM(30, activation='relu', return_sequences=True))
# model.add(LSTM(10, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='relu', return_sequences=True))

model.summary()

# Train
from keras.optimizers import Adagrad, Adam, RMSprop
# optimizer = Adagrad(lr=0.24)
# optimizer = Adam(lr=0.025)
optimizer = RMSprop(lr=0.005)

model.compile(loss='mse', optimizer=optimizer, metrics=['acc','mse'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='mse', patience=50, mode='auto')
history = model.fit(x_train, y_train, epochs=100, batch_size=20, verbose=2,
                     validation_data=(x_test, y_test), callbacks=[early_stopping])

# Predict & Evaluate
y_predict = model.predict(x_test)
print('y_predict(x_test) : \n', y_predict)

loss, _, _ = model.evaluate(x_test, y_test, batch_size=10)
print('mse : ', loss)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test[0], y_predict[0])) 
print("RMSE : ", RMSE(y_test, y_predict))

# Result
pred_data = (df[3108:3113])
# print(pred_data)
pred_data = np.array(pred_data)
# print(pred_data)
pred_data = pred_data.reshape(-1, 5,8)
prediction = model.predict(pred_data)
# print(prediction)

# Save Result
prediction = np.array(prediction)
prediction = np.around(prediction)
# print(prediction.shape)
prediction = prediction.reshape(5, 8)
print(prediction)
np.savetxt("test0822_sjh.csv", prediction, fmt='%d', delimiter=",")