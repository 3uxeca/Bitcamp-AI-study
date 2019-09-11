import keras
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model, Sequential, load_model
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
model_im = load_model('./myself/model_im.h5')


# 2-2. modeling with Functional api(label)
model_la = load_model('./myself/model_la.h5')

# 4. predict
pred_im = model_im.predict(x_test)
pred_la = model_la.predict(x_test)
# print(pred.shape) # (10000, 28, 28, 1)
# print(pred[10])

# n = random.randrange(10000)
# plt.imshow(pred_im[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

fig = plt.figure(figsize=((20, 4)))

for i in range(10):
    n = random.randrange(10000)
    img = pred_im[n].reshape(28,28)
    plot = fig.add_subplot(1, 10, i + 1)
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plot.get_xaxis().set_visible(False)
    plot.get_yaxis().set_visible(False)
    plot.set_title("label : " + str(model_la.predict_classes(x_test[n].reshape((1, 28, 28, 1)))))
    # print('The Answer is ', model_la.predict_classes(x_test[n].reshape((1, 28, 28, 1))))

plt.show()


