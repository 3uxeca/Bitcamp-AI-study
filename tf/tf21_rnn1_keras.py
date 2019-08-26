import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 데이터 구축

idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype=np.str).reshape(-1, 1)
# print(_data.shape) # (7, 1)
# print(_data)
# print(_data.dtype) # <U1 

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype('float32')

x_data = _data[:6,]
y_data = _data[1:,]
# y_data = np.argmax(y_data, axis=1)
# print(x_data)
# print(y_data)
x_data = x_data.reshape(1, 6, 5)
y_data = y_data.reshape(1, 6, 5)
# print(x_data, x_data.shape) # (1, 6, 5)
# print(y_data, y_data.shape) # (1, 6, 5)

#2. 모델 구성  # LSTM의 기본적인 모양 input_shape(/행무시/열,몇개씩자를거야)
model = Sequential()
model.add(LSTM(30,input_shape=(6,5),return_sequences=True))
model.add(LSTM(10,return_sequences=True))
# model.add(Dense(100,activation="relu"))

model.add(LSTM(5,activation="softmax",return_sequences=True))

model.summary()

#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_data, y_data, epochs=500, batch_size=10, verbose=2)

#4. 평가 예측

print("\n test acc: %.4f"%(model.evaluate(x_data, y_data)[1]))
y_predict = model.predict(x_data)

# decoding
y_data = np.argmax(y_data, axis=2)
y_predict = np.argmax(y_predict, axis=2)

result_str = [idx2char[c] for c in np.squeeze(y_predict)]
print("\nPrediction str : ",''.join(result_str))

