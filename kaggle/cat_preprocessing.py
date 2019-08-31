import numpy as np
import pandas as pd

train = np.load('cat_train.npy')
test = np.load('cat_test.npy')

# print(train.shape) # (300000, 25)
# print(test.shape) # (200000, 24)

# print(train[:5])
x_train = train[:, 0:-1]
y_train = train[:, [-1]]
print(x_train.shape, y_train.shape)
print(x_train[:10])
print(y_train[:10])

