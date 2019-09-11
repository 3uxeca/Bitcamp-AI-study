import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-2. data preprocessing
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# print(x_train.shape) # (60000, 28, 28, 1)
# print(x_test.shape) # (10000, 28, 28, 1)
# print(y_train[:10]) # (60000, )
# print(y_test[:10]) # (10000, )

# 1-3. validation data split
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, shuffle=True )

# print(x_train.shape) # (48000, 28, 28, 1)
# print(y_train.shape) # (48000, 10)
# print(x_val.shape) # (12000, 28, 28, 1)
# print(y_val.shape) # (12000, 10)

# 2. modeling with Functional api
inputs = Input(shape=(28,28,1))
x = Conv2D(128, (3,3), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(32, (3,3), activation='relu')(x)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. train
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, verbose=1, batch_size=50)

# 4. model evaluation
loss, acc = model.evaluate(x_test, y_test, verbose=1)

# 5. predict

prediction = model.predict(x_test)

print('loss : ', loss)
print('acc : ', acc)
