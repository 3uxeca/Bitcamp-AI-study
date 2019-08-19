#################### 데 이 터 ####################
from keras.datasets import cifar10
from keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt
import numpy as np


# CIFAR_10은 3채널로 구성된 32x32 이미지 60000장을 갖는다.
IMG_CHANNELS = 3 # input.shape = (32,32,3)
IMG_ROWS = 32
IMG_COLS = 32

## 상수 정의
# BATCH_SIZE = 128
BATCH_SIZE = 800
NB_EPOCH = 20
NB_CLASSES = 10
VERBOSE = 2
# VERBOSE = 1
VALIDATION_SPLIT = 0.2 # 5만개의 데이터중 4만개의 학습데이터와 1만개의 검증데이터로 나누어짐
OPTIM = RMSprop()

(x_train, _), (x_test, _) = cifar10.load_data() # 비지도학습을 위해 y는 뺀다.

x_train = x_train.astype('float32') / 255. # 전처리
x_test = x_test.astype('float32') / 255.

# # 사진 한 장 뽑기!
# pic = x_train[44] # 6만가지의 이미지 중 원하는 n번째 이미지를 보여줘라.
# plt.imshow(pic, cmap=plt.cm.binary)
# plt.show()

# 모양 맞추기
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:],3))) # np.prod() : 각 배열 요소들을 곱하는 함수
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:],3)))
x_train = x_train.reshape(len(x_train), 32, 32, 3)
x_test = x_test.reshape(len(x_test), 32, 32, 3)
print(x_train.shape) # (50000, 32*32*3(3072))
print(x_test.shape) # (10000, 32*32*3(3072))


#################### 모 델 구 성 ####################
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import Regularizer

# 인코딩 될 표현(representation)의 크기
encoding_dim = 32

# 입력 플레이스홀더 # 함수형모델 이용
# input_img = Input(shape=(3072,))
input_img = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
x = Dense(20, activation='relu')(input_img)
# "encoded"는 입력의 인코딩된 표현
encoded = Dense(50, activation='relu')(input_img) # 히든레이어
# "decoded"는 입력의 손실있는 재구성 (lossy reconstruction)
decoded = Dense(3, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='relu')(encoded)

# 입력을 입력의 재구성으로 매핑할 모델
autoencoder = Model(input_img, decoded)         # 784 -> 32 -> 784 # Model(입력값, 출력값)

# 이 모델은 입력을 입력의 인코딩된 입력의 표현으로 매핑
# encoder = Model(input_img, encoded)             # 784 -> 32

# # 인코딩된 입력을 위한 플레이스 홀더
# encoded_input = Input(shape=(encoding_dim,)) # 히든레이어를 decoding의 인풋레이어로 쓰겠다.
# # 오토인코더 모델의 마지막 레이어 얻기
# decoder_layer = autoencoder.layers[-1] # 아웃풋레이어 찾기. (위의 decoded 레이어)
# # 디코더 모델 생성
# decoder = Model(encoded_input, decoder_layer(encoded_input)) # 32 -> 784

autoencoder.summary()
# encoder.summary()
# decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])

history = autoencoder.fit(x_train, x_train, # x_train을 넣어서 x_train을 맞추게 한다.
                          epochs=1, batch_size=256,
                          shuffle=True, validation_data=(x_test, x_test))

# 숫자들을 인코딩 / 디코딩
# test set에서 숫자들을 가져왔다는 것을 유의
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
autoencoder_imgs = autoencoder.predict(x_test)

# print(encoded_imgs)
# print(decoded_imgs)
# print(encoded_imgs.shape) # (10000, 256)
# print(decoded_imgs.shape) # (10000, 3072)


#################### 이 미 지 출 력 ####################
# Matplotlib 사용
import matplotlib.pyplot as plt

n = 10 # 몇 개의 숫자를 나타낼 것인지
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(IMG_ROWS, IMG_COLS, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(autoencoder_imgs[i].reshape(IMG_ROWS, IMG_COLS, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#################### 그 래 프 출 력 ####################
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

loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)
