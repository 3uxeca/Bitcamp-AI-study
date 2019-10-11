import keras
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.utils import np_utils
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, UpSampling2D
from keras.models import load_model, Model, Input, model_from_json, Sequential
from PIL import Image
import numpy as np 
import os

origin_img = Image.open("D:/BITProjects/sjh/imweb/faceapp/media/images/2019/10/10/127-02.jpg")
origin_img = origin_img.resize((64,64))
origin_img = np.array(origin_img)
origin_img = origin_img.reshape(1, 64, 64, 3)
print(origin_img.shape) # 64, 64, 3

inputs = Input(shape=(64,64,3))

x = Conv2D(16, (2,2), padding='same', activation='relu')(inputs)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (2,2), padding='same', activation='relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8, (2,2), padding='same', activation='relu')(x)
x = MaxPooling2D((2,2), padding='same')(x)

x = Conv2D(8, (2,2), padding='same', activation='relu')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(8, (2,2), padding='same', activation='relu')(x)
x = UpSampling2D((2,2))(x)
x = Conv2D(16, (2,2), padding='same', activation='relu')(x)
x = UpSampling2D((2,2))(x)

outputs = Conv2D(3, (2,2), activation='relu', padding='same')(x)

model = Model(inputs, outputs)

model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(origin_img, origin_img, epochs=10, batch_size=8)