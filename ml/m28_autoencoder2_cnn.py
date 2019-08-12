#################### 데 이 터 ####################
from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data() # 비지도학습을 위해 y는 뺀다.

x_train = x_train.reshape(x_train.shape[0], 28 ,28 ,1).astype('float32') / 255 # 6만행(무시) 나머지는 아래 input_shape값이 된다.
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255 # 0~1 사이로 수렴(minmax)시키기 위해 minmaxscaler같은거 필요없이 각 픽셀당 255의 값을 나누어서 데이터 전처리를 하는 과정
print(x_train.shape)
print(x_test.shape)
# print(x_train[:100])


#################### 모 델 구 성 ####################
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.regularizers import Regularizer
from keras.callbacks import EarlyStopping

# 인코딩 될 표현(representation)의 크기
encoding_dim = 32

# 입력 플레이스홀더 # 함수형모델 이용
input_img = Input(shape=(28,28,1))
# input_img = Input(Conv2D(784, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
# x = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2,2), padding='same')(x)
# x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
# x = MaxPooling2D((2,2), padding='same')(x)
# x = Conv2D(8, (3,3), activation='relu', padding='same')(x)

# "encoded"는 입력의 인코딩된 표현
encoded = Conv2D(1, (28,28), padding='same')(input_img)
# encoded = Dense(encoding_dim, (3, 3), activation='relu')(input_img) # 히든레이어
# x = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
# x = UpSampling2D((2,2))(x)
# x = Conv2D(8, (3,3), activation='relu', padding='same')(x)
# x = UpSampling2D((2,2))(x)
# x = Conv2D(16, (3,3), activation='relu')(x)
# x = UpSampling2D((2,2))(x)
# "decoded"는 입력의 손실있는 재구성 (lossy reconstruction)
decoded = Conv2D(1, (28,28), activation='sigmoid', padding='same')(encoded)

# 입력을 입력의 재구성으로 매핑할 모델
autoencoder = Model(input_img, decoded)         # 784 -> 32 -> 784 # Model(입력값, 출력값)

# 이 모델은 입력을 입력의 인코딩된 입력의 표현으로 매핑
encoder = Model(input_img, encoded)             # 784 -> 32

# 인코딩된 입력을 위한 플레이스 홀더
# encoded_input = Input(shape=(encoding_dim,))
encoded_input = Input(shape=(28,28,1)) # 히든레이어를 decoding의 인풋레이어로 쓰겠다.
# 오토인코더 모델의 마지막 레이어 얻기
decoder_layer = autoencoder.layers[-1] # 아웃풋레이어 찾기. (위의 decoded 레이어)
# 디코더 모델 생성
decoder = Model(encoded_input, decoder_layer(encoded_input)) # 32 -> 784

autoencoder.summary()
encoder.summary()
decoder.summary()

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

history = autoencoder.fit(x_train, x_train, # x_train을 넣어서 x_train을 맞추게 한다.
                          epochs=50, batch_size=256,
                          shuffle=True, validation_data=(x_test, x_test))

# 숫자들을 인코딩 / 디코딩
# test set에서 숫자들을 가져왔다는 것을 유의
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs)
print(decoded_imgs)
print(encoded_imgs.shape) # (10000, 32)
print(decoded_imgs.shape) # (10000, 784)


#################### 이 미 지 출 력 ####################
# Matplotlib 사용
import matplotlib.pyplot as plt

n = 10 # 몇 개의 숫자를 나타낼 것인지
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 재구성된 데이터
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
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