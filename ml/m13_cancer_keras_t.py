from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

cancer = load_breast_cancer() # 분류

X = cancer.data
y = cancer.target

# print(X.shape) # (569, 30)
# print(y.shape) # (569,)

# 정규화
# scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X[:100])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# print(X_train.shape) # (398,30)
# print(X_test.shape) # (171, 30)
# print(y_train.shape) # (398,)
# print(y_test.shape) # (171,)

# 모델링
model = Sequential()
model.add(Dense(32, input_dim=30, activation='relu'))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1, activation = 'softmax'))

model.summary()

# 학습
model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=100)

# 평가하기
loss, acc = model.evaluate(X_test, y_test, batch_size=1)
y_predict = model.predict(X_test)

# # decoding
# y_predict = np.argmax(y_predict, axis=1).reshape(-1) # one-hot encoding한 값을 decoding하는 것 [0,0,1])=>[0,1,2]
# y_predict = label_encoder.inverse_transform(y_predict) # int로 encoding했던 str을 다시 불러오는 것

print('loss : ', loss)
print('acc : ', acc)
# print('y_predict(X_test) : \n', y_predict)