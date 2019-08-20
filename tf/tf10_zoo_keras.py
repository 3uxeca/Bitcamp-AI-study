import pandas as pd
import numpy as np
from keras.utils import np_utils


# data
xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1] 
y_data = xy[:, [-1]]

y_data = np_utils.to_categorical(y_data)


# print(x_data.shape)
# print(y_data.shape)


from keras.models import Sequential
from keras.layers import Dense


model = Sequential()
model.add(Dense(7, activation = 'softmax', input_dim = 16))


model.compile(optimizer = 'rmsprop', loss= 'categorical_crossentropy', metrics=['accuracy'])


model.fit(x_data, y_data, epochs=200, batch_size=1)


loss, acc = model.evaluate(x_data, y_data)


print('test_acc : ', acc)