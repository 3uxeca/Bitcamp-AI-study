#-*- coding: utf-8 -*-

from keras.applications import VGG16
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import cv2

# # VGG16 불러오기
# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3)) # 이미지 크기 조절 가능!
# # conv_base = VGG16() # include_top=True, input_shape=(224, 224, 3)가 default 값
# conv_base.summary()
def pretrained_model(img_shape, num_classes):
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
    for layer in conv_base.layers:
        layer.trainable = False
        
        # input format 생성
        keras_input = Input(shape=img_shape, name='image_input')

        # 생성된 모델 사용
        output_conv_base = conv_base(keras_input)
        
        # 레이어 추가 
        x = Flatten(name='flatten')(output_conv_base)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dense(64, activation='relu', name='fc2')(x)
        x = Dense(num_classes, activation='softmax', name='predictions')(x)
        
        # 내가 사용할 모델 만들기
        pretrained_model = Model(inputs=keras_input, outputs=x)
        pretrained_model.compile(loss='sparse_categorical_crossentropy',
                                 optimizer='adam', metrics=['accuracy'])

        return pretrained_model


# MNIST 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# MNIST 데이터를 RGB로 변환하기
X_train = [cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR) for i in X_train]
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')
print(X_train.shape) # (60000, 32, 32, 3)

X_test = [cv2.cvtColor(cv2.resize(i, (32,32)), cv2.COLOR_GRAY2BGR) for i in X_test]
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')
print(X_test.shape) # (10000, 32, 32, 3)

# 학습
model = pretrained_model(X_train.shape[1:], len(set(Y_train)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 최적화 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

#모델의 실행
model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
          epochs=100, batch_size=200, verbose=1,
          callbacks=[early_stopping_callback])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
# 분류모델에서는 accuracy가 정확하다.(회귀모델에서는 mse나 R2를 사용했었음.)


'''
# X_train = X_train.reshape(X_train.shape[0], 32 ,32 ,3).astype('float32') / 255 # 6만행(무시) 나머지는 아래 input_shape값이 된다.
# X_test = X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32') / 255 # 0~1 사이로 수렴(minmax)시키기 위해 minmaxscaler같은거 필요없이 각 픽셀당 255의 값을 나누어서 데이터 전처리를 하는 과정
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_train = np_utils.to_categorical(Y_train) # One Hot Incoding으로 데이터를 변환시킨다. 분류
Y_test = np_utils.to_categorical(Y_test)

# print(X_train.shape) # (60000, 28, 28, 1)
# print(X_test.shape) # (10000, 28, 28, 1)

# 컨볼루션 신경망의 설정
model = Sequential()
# model.add(conv_base)
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28, 28, 1), activation='relu'))
# model.add(Conv2D(32, kernel_size=(3,3), input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # CNN 분류모델에서는 마지막 activation은 분류모델이라는 것을 인식시켜주기 위해서 softmax여야한다. # 0~9까지 10개의 데이터를 보내줄게

model.compile(loss='categorical_crossentropy', # 분류모델에선 loss='mse' 대신 이걸 쓴다!
              optimizer='adam',
              metrics=['accuracy'])

# 모델 최적화 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

#모델의 실행
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                     epochs=30, batch_size=200, verbose=1,
                     callbacks=[early_stopping_callback])

# 테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))
# 분류모델에서는 accuracy가 정확하다.(회귀모델에서는 mse나 R2를 사용했었음.)
'''
