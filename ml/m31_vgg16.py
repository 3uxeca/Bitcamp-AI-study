from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3)) # 이미지 크기 조절 가능!
# conv_base = VGG16() # include_top=True, input_shape=(224, 224, 3)가 default 값
conv_base.summary()

model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.summary()


'''
Dense 256 , activation='relu'
Dense 1 , activation='sigmoid'
summary()
'''
