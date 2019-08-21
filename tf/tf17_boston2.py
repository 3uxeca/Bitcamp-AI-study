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

# y_data = y_data.reshape(y_data.shape[0], 1)
# # print(y_data.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
# print(x_data[:10])

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(
#                                     x_data, y_data, test_size=0.2)

# print(x_train.shape) # (404, 13)
# print(x_test.shape) # (102, 13)
# print(y_train.shape) # (404, 1)
# print(y_test.shape) # (102, 1)
# print(y_train[:10])

x = tf.placeholder(tf.float64, shape=(None, 13))
y_true = tf.placeholder(tf.float64, shape=(None))

with tf.name_scope('inference'):
     w = tf.Variable(tf.zeros([1, 13], dtype=tf.float64, name='weights'))
     b = tf.Variable(0, dtype=tf.float64, name='bias')
     y_pred = tf.matmul(w, tf.transpose(x)) + b
      
with tf.name_scope('loss'):
     loss = tf.reduce_mean(tf.square(y_true-y_pred))  # MSE
      
with tf.name_scope('train'):
     learning_rate = 0.1
     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
     train = optimizer.minimize(loss)
      
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(2001):
        MSE, _ = sess.run([loss, train], feed_dict={x: x_data, 
                                                      y_true: y_data})
      
        if (step+1) % 100 == 0:
            print('Step: {:2d}\t MSE: {:.5f}'.format(step+1, MSE))
