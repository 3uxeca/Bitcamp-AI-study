# boston.npy를 가지고 tensorflow 코딩을 하시오 (선형회귀방식)
# test와 train 분리할 것
# dropout, get_variable, multi layer 등 배운 것을 모두 사용할 것
# R2, RMSE 넣기
import tensorflow as tf
import numpy as np
import random

tf.set_random_seed(777)

x_data = np.load("./data/boston_x.npy", allow_pickle = True)
y_data = np.load("./data/boston_y.npy", allow_pickle = True)
# print(x_data.shape) # (506, 13)
# print(y_data.shape) # (506, 1)
# print(x_data[:10])
# print(y_data[:10])

y_data = y_data.reshape(y_data.shape[0], 1)
# print(y_data.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
# print(x_data[:10])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
                                    x_data, y_data, test_size=0.2)

# print(x_train.shape) # (404, 13)
# print(x_test.shape) # (102, 13)
# print(y_train.shape) # (404, 1)
# print(y_test.shape) # (102, 1)
# print(y_train[:10])

keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, shape=[None, 13])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W1 = tf.get_variable("W1", shape=[13, 64], initializer=tf.contrib.layers.xavier_initializer()) 
b1 = tf.Variable(tf.random_normal([64]))
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1) 
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob) 

W2 = tf.get_variable("W2", shape=[64, 50], initializer=tf.contrib.layers.xavier_initializer()) 
b2 = tf.Variable(tf.random_normal([50]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2) 
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[50, 1], initializer=tf.contrib.layers.xavier_initializer()) 
b3 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(layer2, W3) + b3

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# num_epochs = 15
# batch_size = 100

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([train, cost, accuracy], 
                                        feed_dict={X: x_train, Y: y_train, keep_prob:0.7})
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_test, keep_prob: 1})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape

    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 1}))
