from keras.applications import VGG16
from keras.applications import VGG19, Xception, InceptionV3,ResNet50, MobileNet
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.datasets import cifar10
import numpy as np

## 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
print(x_train.shape, 'train samples')
# print(X_test.shape[0], 'test samples')
print(x_test.shape, 'test samples')

## 범주형으로 변환 (기계가 빨리 번역해서 인식하게 하기 위해서, one hot encoding!!)
y_train = np_utils.to_categorical(y_train)

## 실수형으로 지정하고 정규화
x_train = x_train.astype('float32')
x_train /= 255

# MODEL
conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(32,32,3))
model = Sequential()

model.add(conv_base) # vgg16 (None, 1, 1, 512)

model.add(Flatten()) # flatten_1 (None, 512)
model.add(Dense(1)) # dense_1 (None, 256)

model.add(Dense(10, activation="softmax")) # dense_2 (None, 10)
model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=10, batch_size=2048)
# print("acc: ",model.evaluate(x_test,y_test)[1])

'''
#################### 그 래 프 출 력 ####################
import matplotlib.pyplot as plt
def plot_acc(history, title=None):
    # summarize history for aaccuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validation data'], loc=0)
    # plt.show()

def plot_loss(history, title=None):
    # smmarize history for loss
    if not isinstance(history, dict):
        history = history.history
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training data', 'Validaiton data'], loc=0)
    # plt.show()

plot_acc(history, '(a) 학습 경과에 따른 정확도 변화 추이')
plt.show()
plot_loss(history, '(b) 학습 경과에 따른 손실값 변화 추이')
plt.show()

# print("acc: ",model.evaluate(x_test,y_test)[1])
loss, acc = model.evaluate(x_test, x_test)
print(loss, acc)
'''