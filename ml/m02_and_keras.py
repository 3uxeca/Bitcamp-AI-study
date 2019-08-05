## m02_and.py를 Keras를 이용한 Deep Learning Model로 다시 만들기

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터 
x_data = np.array([[0,0], [1,0], [0,1], [1,1]]) # shape(4,2)
y_data = np.array([0,0,0,1])

x_test = np.array([[0,0], [1,0], [0,1], [1,1]])
y_test = np.array([0,0,0,1])

# 2. 모델
model = Sequential()

model.add(Dense(64, input_dim = 2, activation = 'relu'))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1, activation='sigmoid'))


# 3. 실행
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_data, y_data, epochs=100, batch_size=1, verbose=1)

# 4. 평가 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
y_predict = model.predict_classes(x_test)

print('loss : ', loss)
print('acc : ', acc)
print('y_predict(x_test) : \n', y_predict)

# print(x_test, "의 예측결과 : ", y_predict)
# print("acc : ", acc)) # accuracy_score(원래값, 비교값)