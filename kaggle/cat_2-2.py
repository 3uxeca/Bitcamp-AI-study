import numpy as np
import pandas as pd

X = np.load("X.npy")
y = np.load("y.npy")
submission = pd.read_csv('./kaggle/cat/sample_submission.csv')

# print(X.shape) # (300000, 23)
# print(y.shape) # (300000, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True
)

print(x_train.shape) # (240000, 23)
print(x_test.shape) # (60000, 23)
print(y_train.shape) # (240000, )
print(y_test.shape) # (60000, )

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout

model = Sequential()
model.add(Dense(256, input_dim=23, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=1)

print("\n Accuracy: %.4f" % (model.evaluate(x_test, y_test)[1]))

y_predict = model.predict(x_test)
print('y_predict(X_test) : \n', y_predict)

# print(type(y_predict), y_predict.shape)

submission['target'] = y_predict(x_test)[:,1]
submission.to_csv('submission6.csv', index=False)