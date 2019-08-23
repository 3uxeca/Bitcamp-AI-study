import numpy as np 
import tensorflow as tf
import random
tf.set_random_seed(777)

# Iris Data
x_data = np.load("./data/boston_x.npy")
y_data = np.load("./data/boston_y.npy")

# print(x_data.shape) # (506, 13)
# print(y_data.shape) # (506, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, shuffle=True)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
print(x_train.shape) # (404, 13)
print(x_test.shape) # (102, 13)
print(y_train.shape) # (404, 1)
print(y_test.shape) # (102, 1)

# input place holders
X = tf.placeholder(tf.float32, [None, 13])
Y = tf.placeholder(tf.float32, [None, 1]) 

# Model
L1 = tf.layers.dense(X, 128, activation=tf.nn.relu)
L2 = tf.layers.dense(L1, 128, activation=tf.nn.relu)
L3 = tf.layers.dense(L2, 84, activation=tf.nn.relu)
L4 = tf.layers.dense(L3, 64, activation=tf.nn.relu)
L5 = tf.layers.dense(L4, 32, activation=tf.nn.relu)

hypothesis = tf.layers.dense(L5, 1, activation=tf.nn.relu)

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # compile

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) # compile
# train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

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
                                        feed_dict={X: x_train, Y: y_train})
        if step % 200 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_test})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape

    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
