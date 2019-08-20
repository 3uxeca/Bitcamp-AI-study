from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.models import Model
import numpy as np
import os 
import tensorflow as tf

# data
xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1] 
y_data = xy[:, [-1]]

# print(x_data.shape, y_data.shape) # x.shape(101, 16) / y.shape(101,1)

y_data = np_utils.to_categorical(y_data)

# print(x_data.shape, y_data.shape) # x.shape(101, 16) / y.shape(101,7)

# model
model = Sequential()
model.add(Dense(7, activation='softmax', input_shape=(16,)))

# train
optimizer = SGD(lr=0.1)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['mse'])

model.fit(x_data, y_data)

# evaluate & predict
mse, _ = model.evaluate(x_data, y_data)
print("mse : ", mse)
pred1 = model.predict(y_data)
print(pred1)
