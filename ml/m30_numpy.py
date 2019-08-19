import numpy as np 
a = np.arange(10)
print(a)
np.save("aaa.npy", a)
b = np.load("aaa.npy")
print(b)

# 모델 저장하기 ##
model.save('savetest01.h5')
## 모델 불러오기 ##
from keras.models import load_model
model = load_model('savetest01.h5')
from keras.layers import Dense
model.add(Dense(1))

## pandas를 numpy로 바꾸기 ###
pandas.value

## csv 불러오기 ###
dataset = numpy.loadtxt('./data/pima-indians-diabetes.csv', delimiter=',')
iris_data = pd.read_csv('./data/iris.csv', encoding='utf-8')
            # index_col = 0, encoding='cp949', sep=';', encoding='utf-8')
            # names = ['x1', 'x2', 'x3', 'x4', 'y']
 wine = pd.read_csv("./data/winequality-white.csv", sep=",", encoding="utf-8")


## utf-8 ###
-*- coding: utf-8 -*-


## 한글처리 ###


## 각종 샘플 데이터 셋 ###
from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data() # keras로 땡겨오면 바로 numpy로 들어옴

from keras.datasets import boston_housing
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

from sklearn.datasets import load_boston
boston = load_boston()
print(boston.keys()) # sklearn으로 땡겨오면 data(x)와 target(y)로 나누어짐
# boston.data : x값, numpy type
# boston.target : y값, numpy type

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
