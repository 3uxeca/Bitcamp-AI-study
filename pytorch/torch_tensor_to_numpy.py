import torch
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from keras.layers import Input, Dense, Conv2D, Lambda, concatenate, MaxPool2D, Reshape, Flatten, UpSampling2D
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras import backend as K
from keras import objectives
from keras.losses import mse, kullback_leibler_divergence

# a = torch.ones(5,5)
# # print(a)

# b = a.numpy()
# # print(b)

# a.add_(1)
# print(a)
# print(b)

# a = np.ones(5)
# b = torch.from_numpy(a)
# # c = np.zeros(5)
# np.add(a, 1, out=a)
# print(a)
# print(b)

# x = torch.ones(5)

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     y = torch.ones_like(x, device=device)
#     x = x.to(device)
#     z = x + y
#     print(z)
#     print(z.to("cpu", torch.double))

#####################################################################################

# parameters
batch_size = 32
epoch = 200
z_dim = 2
img_size = 64
# img_size = x_train.shape[2]
n_labels = 13
color = 3
patience = 10
n_hidden = 32
line = 37

# load data
x = np.load("./sjh/npy/cvaex_origin_image_sample_{}_c_crop.npy".format(str(img_size))) # 정면을 포함한 각도별 10개의 이미지
y = np.load("./sjh/npy/cvaey_origin_image_sample_{}_c_crop.npy".format(str(img_size))) # 각 각도의 라벨링 0~9
# test = np.load("./npy/test.npy")
x = x.reshape(x.shape[0],x.shape[1],x.shape[2], color)
x = x.astype("float32") / 255
y = to_categorical(y, num_classes= n_labels)
print(x.shape)
print(y.shape)
x_train = x[:180 * n_labels]
y_train = y[:180 * n_labels]
print(x_train.shape)
print(y_train.shape)
x_test = x[180 * n_labels:]
y_test = y[180 * n_labels:]
print(x_test.shape)
print(y_test.shape)