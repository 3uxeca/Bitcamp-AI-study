import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16;
from keras.applications.vgg16 import preprocess_input
from keras.datasets import mnist


# MNIST 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# print(X_train.shape) # (60000, 28, 28)
# print(X_test.shape) # (10000, 28, 28)
# print(Y_train.shape) # (60000,)
# print(Y_test.shape) # (10000,)

X_train = X_train.reshape(len(X_train), np.prod(X_train.shape[1:]))
X_test = X_test.reshape(len(X_test), np.prod(X_test.shape[1:]))
# print(X_train.shape) # (60000, 784)
# print(X_test.shape) # (10000, 784)

classes = np.unique(Y_train)
num_classes = 10

# 3채널 이미지로 변경
X_train = np.dstack([X_train] * 3)
X_test = np.dstack([X_test] * 3)
# print(X_train.shape) # (60000, 784, 3)
# print(X_test.shape) # (10000, 784, 3)

# 4차원 이미지로 변경
X_train = X_train.reshape(-1, 28, 28, 3)
X_test = X_test.reshape(-1, 28, 28, 3)
# print(X_train.shape) # (60000, 28, 28, 3)
# print(X_test.shape) # (10000, 28, 28, 3)

# # VGG16을 위한 이미지 사이즈 변경 -> 48 x 48
from keras.preprocessing.image import img_to_array, array_to_img
X_train = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in X_train])
X_test = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in X_test])
# print(X_train.shape) # (60000, 48, 48, 3)
# print(X_test.shape) # (10000, 48, 48, 3)

# 정규화, 데이터 유형 변경
X_train = X_train / 255.
X_test = X_test / 255.
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Y데이터 One-Hot-Encoding
Y_train_ohe = to_categorical(Y_train)
Y_test_ohe = to_categorical(Y_test)

# 훈련데이터와 검증데이터 나누기
X_train,X_val,Y_train,Y_val = train_test_split(X_train, Y_train_ohe,
                                               test_size=0.2, random_state=13)

# print(X_train.shape) # (48000, 48, 48, 3)
# print(X_val.shape) # (12000, 48, 48, 3)
# print(Y_train.shape) # (48000, 10)
# print(Y_val.shape) # (12000, 10)

# VGG16 모델을 위한 상수 정의 
IMG_WIDTH = 48
IMG_HEIGHT = 48
IMG_DEPTH = 3
BATCH_SIZE = 16

# Input 전처리
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test  = preprocess_input (X_test)

# VGG16 모델 생성
conv_base = VGG16(weights='imagenet',
                  include_top=False, 
                  input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

# conv_base.summary()

# 특성 추출
train_features = conv_base.predict(np.array(X_train), batch_size=BATCH_SIZE, verbose=1)
test_features = conv_base.predict(np.array(X_test), batch_size=BATCH_SIZE, verbose=1)
val_features = conv_base.predict(np.array(X_val), batch_size=BATCH_SIZE, verbose=1)
#for layer in conv_base.layers:
#    layer.trainable = False

# 6.1 Saving the features so that they can be used for future
np.savez("train_features", train_features, Y_train)
np.savez("test_features", test_features, Y_test)
np.savez("val_features", val_features, Y_val)

# Current shape of features
print(train_features.shape, "\n",  test_features.shape, "\n", val_features.shape)

# Flatten extracted features
train_features_flat = np.reshape(train_features, (48000, 1*1*512))
test_features_flat = np.reshape(test_features, (10000, 1*1*512))
val_features_flat = np.reshape(val_features, (12000, 1*1*512))

from keras import models
from keras.models import Model
from keras import layers
from keras import optimizers
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU

# 7.0 Define the densely connected classifier followed by leakyrelu layer and finally dense layer for the number of classes
NB_TRAIN_SAMPLES = train_features_flat.shape[0]
NB_VALIDATION_SAMPLES = val_features_flat.shape[0]
NB_EPOCHS = 100

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=(1*1*512)))
model.add(layers.LeakyReLU(alpha=0.1))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model.
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(),
  # optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

# Incorporating reduced learning and early stopping for callback
reduce_learning = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='auto',
    epsilon=0.0001,
    cooldown=2,
    min_lr=0)

eary_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=7,
    verbose=1,
    mode='auto')

callbacks = [reduce_learning, eary_stopping]

# Train the the model
history = model.fit(
    train_features_flat,
    Y_train,
    epochs=NB_EPOCHS,
    validation_data=(val_features_flat, Y_val),
    callbacks=callbacks
)

# plot the loss and accuracy

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')

plt.legend()

plt.show()