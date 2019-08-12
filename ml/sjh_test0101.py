from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras import regularizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt 

# CIFAR_10은 3채널로 구성된 32x32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3 # input.shape = (32,32,3)
IMG_ROWS = 32
IMG_COLS = 32

## 상수 정의
# BATCH_SIZE = 128
BATCH_SIZE = 40
NB_EPOCH = 200
NB_CLASSES = 10
VERBOSE = 2
# VERBOSE = 1
VALIDATION_SPLIT = 0.2 # 5만개의 데이터중 4만개의 학습데이터와 1만개의 검증데이터로 나누어짐
OPTIM = RMSprop()

# 데이터셋 불러오기
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# print('X_train shape : ', X_train.shape) #(50000, 32, 32, 3)
# print('X_test shape : ', X_test.shape) # (50000, 32, 32, 3)
# print(X_train.shape[0], 'train samples') # 50000 train samples
# print(X_test.shape[0], 'test samples')  # 10000 test samples

# print('y_train shape : ', y_train.shape) # (50000, 1)
# print('y_train shape : ', y_test.shape) # (50000, 1)
# print(y_train.shape[0], 'train samples') # 50000 train samples
# print(y_test.shape[0], 'test samples')  # 10000 test samples

# 데이터셋 300개로 자르기
X_train = X_train[0:300]
y_train = y_train[0:300]
X_test = X_test[0:300]
y_test = y_test[0:300]

# print(X_train.shape) # (300, 32, 32, 3)
# print(y_train.shape) # (300, 1)
# print(X_test.shape) # (300, 32, 32, 3)
# print(y_test.shape) # (300, 1)

# # 사진 한 장 뽑아보기
# pic = X_train[3] # 300개 이미지 중 n번째 이미지
# plt.imshow(pic, cmap=plt.cm.binary)
# plt.show()

# 실수형으로 지정 후 정규화
# X_train = X_train.reshape(X_train.shape[0], IMG_ROWS * IMG_COLS *IMG_CHANNELS).astype('float32') / 255 
# X_test = X_test.reshape(X_test.shape[0], IMG_ROWS * IMG_COLS *IMG_CHANNELS).astype('float32') / 255 

# 범주형으로 전환
y_train = np_utils.to_categorical(y_train, NB_CLASSES) # One Hot Incoding으로 데이터를 변환시킨다. 분류
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# print(X_train.shape) # (300, 3072)
# print(X_test.shape) # (300, 3072)
# print(y_train.shape) # (300, 10)
# print(y_test.shape) # (300, 10)


# 이미지 생성기 ImageDataGenerator

from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(
    rotation_range=20, # 회전값 범위 20
    width_shift_range=0.02, # 넓이 0.02
    height_shift_range=0.02, # 높이 0.02
    horizontal_flip=True # 수평값
)    

# model.fit_generator(data_generator.flow(X_train, y_train, batch_size=40), # 훈련을 시키되, 발전기도 같이 실행해라!
#                     steps_per_epoch= 300 // 32, # 몇 배로 증폭시킬건지 steps*batch_size 자기 숫자만큼의 새로운 이미지 데이터 생성
#                     epochs=2,
#                     validation_data=(X_test, y_test),
#                     verbose=1) 


# 하이퍼 파라미터 최적화
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def build_network(keep_prob=0.5, optimizer='rmsprop'):
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    model.add(Activation('relu')) 
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))) #
    model.add(Activation('relu'))
    model.add(BatchNormalization()) 
    model.add(MaxPooling2D(pool_size=(2,2)))#
    model.add(Dropout(0.2))

    # model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))) #
    # model.add(Activation('relu'))
    # model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))) #
    # model.add(Activation('relu'))
    # model.add(BatchNormalization()) 
    # model.add(MaxPooling2D(pool_size=(2,2)))#
    # model.add(Dropout(0.3))

    # model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))) #
    # model.add(Activation('relu'))
    # model.add(BatchNormalization()) 
    # model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))) #
    # model.add(Activation('relu'))
    # model.add(BatchNormalization()) 
    # model.add(MaxPooling2D(pool_size=(2,2)))#
    # model.add(Dropout(0.4))

    model.add(Flatten()) # 이하 DNN\
    model.add(Dense(NB_CLASSES)) # NB_CLASSES = 10 = Output 분류모델에서는 무조건 주어진 10개 중 선택해야한다.
    model.add(Activation('softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(data_generator.flow(X_train, y_train, batch_size=40), # 훈련을 시키되, 발전기도 같이 실행해라!
                        steps_per_epoch= 100, # 몇 배로 증폭시킬건지 steps*batch_size 자기 숫자만큼의 새로운 이미지 데이터 생성
                        epochs=2,
                        validation_data=(X_test, y_test),
                        verbose=1) 
    return model

def create_hyperparameters():
    batches = [10, 20, 30, 40, 50]
    optimizers = ['rmsprop', 'adam', 'adadelta'] # 용도에 맞게 쓰자.
    # dropout = np.linspace(0.1, 0.5, 5)
    epochs = [100, 200, 300, 400, 500]
    return{"batch_size":batches, "optimizer":optimizers,"epochs":epochs} #  "keep_prob":dropout

from keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_network, verbose=1) 

hyperparameters = create_hyperparameters()

from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator=model,
                             param_distributions=hyperparameters,
                             n_iter=10, n_jobs=1, cv=3, verbose=1)

search.fit(X_train, y_train) # 데이터 집어넣기!

print(search.best_params_)
