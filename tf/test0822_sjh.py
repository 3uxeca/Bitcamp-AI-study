import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

# Data
df = pd.read_csv("./data/test0822.csv", sep=",")
# print(df[:10])
# print(df.shape) # (5479, 9)

# print(df['kp_0h'].value_counts())

# Train, test data split
df = df.drop("date", 1)
df_train = df.loc[0:3112]
df_test = df.loc[3118:]

# print(df_train) # 1999-01-01 ~ 2007-07-10 기간의 데이터. "date" 열 삭제.
# print(df_test) # 2007-07-16 ~ 2013-12-31 기간의 데이터. "date" 열 삭제.
# print(df_train.shape) # (3313, 8) 
# print(df_test.shape) # (2361, 8)

df_train = df_train.T # 행렬전치
df_test = df_test.T
df_train = np.array(df_train)
df_test = np.array(df_test)
# print(df_train.shape) # (8, 3113)
# print(df_test.shape) # (8, 2361)
# print(df_train[:5])
# print(df_test[:5])


from sklearn.model_selection import train_test_split
x_train, y_train, x_test, y_test = train_test_split(
    df_train, df_test, test_size=0.2, shuffle=False)

# print(type(x_train))
# print(x_train.shape) # (6, 3113)
# print(x_test.shape) # (6, 2361)
# print(y_train.shape) # (2, 3113)
# print(y_test.shape) # (2, 2361)

# # Scaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

size = 6

def x_train_split_6(seq, size): 
    aaa = []
    for i in range(len(x_train)-size + 1): 
        subset1 = x_train[i:(i+size)]
        aaa.append(subset1)
        #aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa) 

def x_test_split_6(seq, size): 
    bbb = []
    for i in range(len(x_test)-size + 1): 
        subset2 = x_test[i:(i+size)]
        bbb.append(subset2)
        #aaa.append([item for item in subset])
    print(type(bbb))
    return np.array(bbb) 

x_train_dataset = x_train_split_6(x_train, size)
x_test_dataset = x_test_split_6(x_test, size)
# # print(x_train_dataset) 
# print(x_train_dataset.shape) # (1, 6, 3113)
# print(x_test_dataset.shape) # (1, 6, 2361)


x_train_dataset = np.transpose(x_train_dataset, axes=(2, 1, 0))
y_train_dataset = np.transpose(y_train, axes=(1, 0))
x_test_dataset = np.transpose(x_test_dataset, axes=(2, 1, 0))
y_test_dataset = np.transpose(y_test, axes=(1, 0))
# print(x_train_dataset.shape) # (3113, 6, 1)
# print(y_train_dataset.shape) # (3113, 2)
# print(x_test_dataset.shape) # (2361, 6, 1)
# print(y_test_dataset.shape) # (2361, 2)

# Model
model = Sequential()

model.add(LSTM(10, input_shape=(6,1), return_sequences=True))
model.add(LSTM(10))

model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(300, activation='relu'))

model.add(Dense(2))

model.summary()

# Train
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x_train_dataset, y_train_dataset, epochs=300, batch_size=300, verbose=1,
          callbacks=[early_stopping])

# Predict
loss, acc = model.evaluate(x_test_dataset, y_test_dataset)

y_predict = model.predict(x_test_dataset)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test_dataset, y_predict): 
    return np.sqrt(mean_squared_error(y_test_dataset, y_predict)) 
print("RMSE : ", RMSE(y_test_dataset, y_predict))
