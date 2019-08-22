import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten

# Data
df = pd.read_csv("./data/test0822.csv", sep=",")
# print(df[:10])
# print(df.shape) # (5479, 9)

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
# print(df_train.shape) # (8, 3313)
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

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
# print(x_train.shape) # (3113, 6)
# print(x_test.shape) # (2361, 6)
# print(y_train.shape) # (3113, 2)
# print(y_test.shape) # (2361, 2)

# 모델의 설정
model = Sequential()
model.add(Dense(64, input_dim=6, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(2))

model.summary()

# 모델 최적화 설정
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=50, mode='auto')

# 학습
model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=200) 
        #   callbacks=[early_stopping] )

# 평가하기
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict(x_test)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): 
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2(결정계수) 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
