from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt 


# Data load
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
#      (60000, 28, 28)  (60000,)   (10000, 28, 28)   (10000,)


X_train = X_train.astype('float32') / 255 
X_test = X_test.astype('float32') / 255 
Y_train = np_utils.to_categorical(Y_train) # One Hot Incoding
Y_test = np_utils.to_categorical(Y_test)

# Modeling
model = Sequential()
model.add(LSTM(40, activation = 'relu', input_shape=(28,28)))
# model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10, activation='softmax'))

model.summary()

# Train & Predict
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(X_train, Y_train, epochs=10, batch_size=100, verbose=1,
          callbacks=[early_stopping]  )

#4. 평가 예측
loss, acc = model.evaluate(X_test, Y_test)

y_predict = model.predict(X_test)

print('loss : ', loss)
print('acc : ', acc)
# print('y_predict(X_test) : \n', y_predict)
