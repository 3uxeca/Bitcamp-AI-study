import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 1-2. data preprocessing
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# 1-3. validation data split
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, shuffle=True )

# 2. modeling with Functional api
model0 = Sequential()


model0.add(Conv2D(128, (3,3), input_shape=(28,28,1), activation='relu'))
model0.add(MaxPooling2D(2,2))
model0.add(Conv2D(64, (3,3), activation='relu'))
model0.add(MaxPooling2D(2,2))
model0.add(Conv2D(32, (3,3), activation='relu'))
model0.add(Flatten())

x = Dense(10, activation='relu')(model0.output)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=model0.input, outputs=outputs)

model_output = Sequential()
model_output.add(model)

model0.summary()
model.summary()


model_output.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. train
model_output.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1, verbose=1, batch_size=50)

# 4. model evaluation
loss, acc = model_output.evaluate(x_test, y_test, verbose=1)

# 5. predict

prediction = model_output.predict(x_test)

# print(prediction.shape) # (10000, 10)
# print(prediction[:10])
# print('loss : ', loss)
# print('acc : ', acc)

# n = 10
# plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()
# print('The Answer is ', model_output.predict_classes(x_test[n].reshape((1, 28, 28, 1))))


for n in range(10):
    plt.imshow(x_test[n].reshape(28,28), cmap='Greys', interpolation='nearest')
    
    plt.show()
    print('The Answer is ', model_output.predict_classes(x_test[n].reshape((1, 28, 28, 1))))
