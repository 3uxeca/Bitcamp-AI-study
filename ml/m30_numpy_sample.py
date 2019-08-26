import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import boston_housing
from sklearn.datasets import load_boston
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
'''
#pima-indians-diabetes
pima = np.loadtxt("./data/pima-indians-diabetes.csv", delimiter=",")
np.save("pima.npy", pima)
pima_load = np.load("pima.npy")
print(pima_load.shape) # (768,9)

# iris
iris = pd.read_csv("./data/iris.csv", encoding='utf-8', names=['a', 'b', 'c', 'd', 'y'], sep=",")
y = iris.loc[:, "y"] # .loc 레이블로 나누기(레이블은 실질적인 데이터가 아님)
x = iris.loc[:,["a", "b", "c", "d"]]

y = np.array(y)
x = np.array(x)

for i in range(len(y)):
    if y[i] == 'Iris-setosa':
        y[i] = 1
    elif y[i] == 'Iris-versicolor':
        y[i] = 2
    else:
        y[i] = 3

np.save("iris_x.npy", x)
np.save("iris_y.npy", y)
iris_load_x = np.load("iris_x.npy")
iris_load_y = np.load("iris_y.npy")
print(iris_load_x.shape) # (150,4)
print(iris_load_y.shape) # (150,)

# wine
wine = pd.read_csv("./data/winequality-white.csv", sep=";")
np.save("wine.npy", wine)
wine_load = np.load("wine.npy")
print(wine_load.shape) # (4898,12)

# MNIST
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
x = np.concatenate((X_train, X_test))
y = np.concatenate((Y_train, Y_test))
np.save("mnist_x.npy", x)
np.save("mnist_y.npy", y)
mnist_x_load = np.load("mnist_x.npy")
mnist_y_load = np.load("mnist_y.npy")
print(mnist_x_load.shape) # (70000, 28, 28)
print(mnist_y_load.shape) # (70000,)

# CIFAR10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
x = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
np.save("cifar_x.npy", x)
np.save("cifar_y.npy", y)
cifar_x_load = np.load("cifar_x.npy")
cifar_y_load = np.load("cifar_y.npy")
print(cifar_x_load.shape) # (60000, 32, 32, 3)
print(cifar_y_load.shape) # (60000, 1)
'''
# boston_housing
# (X_train, y_train), (X_test, y_test) = boston_housing.load_data()
# x = np.concatenate((X_train, X_test))
# y = np.concatenate((y_train, y_test))
# np.save("boston_x.npy", x)
# np.save("boston_y.npy", y)
# boston_load_x = np.load("boston_x.npy")
# boston_load_y = np.load("boston_y.npy")
# print(boston_load_x.shape) # (506, 13)
# print(boston_load_y.shape) # (506,)
# print(boston_load_x[:10])

# from keras.datasets import boston_housing
# (x_train, y_train),(x_test,y_test) = boston_housing.load_data()
# boston_housing_x = np.vstack((np.array(x_train), np.array(x_test)))
# boston_housing_y = np.hstack((np.array(y_train), np.array(y_test)))
# np.save("boston_housing_x.npy",boston_housing_x)
# np.save("boston_housing_y.npy",boston_housing_y)
# x = np.load("boston_housing_x.npy")
# y = np.load("boston_housing_y.npy")
# # mnist_train =
# print("boston_housingx",x.shape)
# print("boston_housingy",y.shape)


# breast_cancer
cancer = load_breast_cancer()
cancer = np.c_[cancer.data,cancer.target]
np.save("cancer.npy", cancer)
cancer_load = np.load("cancer.npy")
print(cancer_load.shape) # (569,31)

### 한글처리 찾기
### csv 저장하는 법 찾기
# from sklearn.datasets import load_breast_cancer
# cancer = load_breast_cancer()
# x = np.array(cancer["data"])
# y = np.array(cancer["target"])


# np.save("cancer_x.npy", x)
# cancer_x = np.load("cancer_x.npy")
# print(cancer_x.shape)


# np.save("cancer_y.npy", y)
# cancer_y = np.load("cancer_y.npy")
# print(cancer_y.shape)