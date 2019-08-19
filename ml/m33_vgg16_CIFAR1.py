from keras.applications import VGG16
from keras.layers import *
from keras.models import *
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

(x_train, y_train) , (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')/255
# x_trina = x_train[:1000]
# x_test = x_test.astype('float32')/255
y_train = to_categorical(y_train)
# y_train = y_train[:1000]
# y_test = to_categorical(y_test)


conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape=(32,32,3))
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ["accuracy"])

history= model.fit(x_train, y_train, epochs = 1, batch_size = 2048, shuffle = True)