import pandas as pd
import numpy as np 
train=pd.read_csv('./kaggle/cat/train.csv')
test=pd.read_csv('./kaggle/cat/test.csv')
 
np.save("cat_train.npy", train)
np.save("cat_test.npy", test)

train_load = np.load("cat_train.npy")
print(train_load.shape)