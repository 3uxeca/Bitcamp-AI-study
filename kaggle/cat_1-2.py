import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
import xgboost
from sklearn.metrics import roc_auc_score

train = np.load("train.npy")
test = np.load("test.npy")
# print(train)
# print(test)
# print(train.shape) # (300000, 23)
# print(test.shape) # (200000, 23)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    train, test, test_size=0.2, shuffle=False
)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)