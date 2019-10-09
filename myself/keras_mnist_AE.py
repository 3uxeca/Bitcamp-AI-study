import keras
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-2. data preprocessing
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(x_train.shape) # (60000, 28, 28, 1)
# print(x_test.shape) # (10000, 28, 28, 1)

# 1-3. validation data split
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, shuffle=True )

# 2. modeling with Functional api(image)
inputs_im = Input(shape=(28,28,1))

x = Conv2D(16, (3,3), padding='same', activation='relu')(inputs_im)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
encoded = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8, (3,3), padding='same', activation='relu')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (3,3), padding='same', activation='relu')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (3,3), activation='relu')(x)
x = UpSampling2D((2,2))(x)


decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same')(x)

autoencoder = Model(inputs_im, decoded)

model_im = Sequential()
model_im.add(autoencoder)
# model.summary()

model_im.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# 2-2. modeling with Functional api(label)
inputs_la = Input(shape=(28,28,1))
x = Conv2D(128, (3,3), activation='relu')(inputs_la)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(32, (3,3), activation='relu')(x)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(10, activation='relu')(x)

model2 = Model(inputs=inputs_la, outputs=outputs)

model_la = Sequential()
model_la.add(model2)
model_la.add(Dense(10, activation='softmax'))

# model.summary()
model_la.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 3. train & evaluate
model_im.fit(x_train, x_train, epochs=50, batch_size=50, validation_data=(x_test,x_test))
model_la.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, verbose=1, batch_size=50)
model_im.save('./myself/model_im.h5')
model_la.save('./myself/model_la.h5')

model_im.evaluate(x_test, x_test)
model_la.evaluate(x_test, y_test)


'''
# 4. predict
pred_im = model_im.predict(x_test)
pred_la = model_la.predict(x_test)
# print(pred.shape) # (10000, 28, 28, 1)
# print(pred[10])

# n = random.randrange(10000)
# plt.imshow(pred_im[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

fig = plt.figure()

for i in range(10):
    n = random.randrange(10000)
    img = pred_im[n].reshape(28,28)
    plot = fig.add_subplot(1, 10, i + 1)
    plt.imshow(img, cmap='Greys')
plt.show()

# print('The Answer is ', model_la.predict_classes(x_test[n].reshape((1, 28, 28, 1))))

'''
