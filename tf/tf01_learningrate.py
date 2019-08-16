#1. 데이터
import numpy as np 
x = np.array([1,2,3,4])
y = np.array([1,2,3,4])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense 
model = Sequential()

model.add(Dense(5, input_dim = 1, activation = 'relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. 훈련
from keras.optimizers import Adam, Adadelta, Adagrad, SGD, Nadam, Adamax, RMSprop
# optimizer = Adam(lr=0.025)
# optimizer = Adadelta(lr=0.933)
# optimizer = Adagrad(lr=0.2)
# optimizer = Nadam(lr=0.0035)
# optimizer = SGD(lr=0.01)
# optimizer = Adamax(lr=0.09)
optimizer = RMSprop(lr=0.005)



model.compile(loss = 'mse', optimizer = optimizer, metrics = ['mse'])
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

model.fit(x, y, epochs=100, batch_size = 1)

#4. 평가 예측
mse, _ = model.evaluate(x, y, batch_size=1)
print("mse : ", mse)
pred1 = model.predict([1.5, 2.5, 3.5])
print(pred1)

'''
# optimizer = Adam(lr=0.025)
=> mse :  8.526512829121202e-14
=> [[1.5000004]
    [2.5000005]
    [3.4999998]]
# optimizer = Adadelta(lr=0.933) 
=> mse : 0.00021375643791543553 
=> [[1.5182717]
    [2.5066054]
    [3.4949386]]
# optimizer = Adagrad(lr=0.2)
=> mse :  7.105427357601002e-15
=> [[1.4999999]
    [2.4999998]
    [3.5      ]]
# optimizer = Nadam(lr=0.0035)
=> mse :  1.4848215883489502e-05
=> [[1.504833 ]
    [2.5017726]
    [3.498712 ]]
# optimizer = SGD(lr=0.01)
=> mse :  1.324821141679422e-09 
=> [[1.5000418]
    [2.500036 ]
    [3.5000298]]
# optimizer = Adamax(lr=0.09)
=> mse :  0.0
=> [[1.5000001]
    [2.5      ]
    [3.5000002]]
# optimizer = RMSprop(lr=0.005)
=> mse :  3.6764655249044154e-06
=> [[1.5025488]
    [2.5012434]
    [3.4999378]]
'''