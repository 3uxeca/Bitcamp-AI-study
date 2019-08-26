import numpy as np 
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

tf.set_random_seed(77)

# hyper parameters
learning_rate = 0.00001

'''
# Iris Data
x_data = np.load("./data/cancer_x.npy")
y_data = np.load("./data/cancer_y.npy")

# print(x_data.shape) # (569, 30)
# print(y_data.shape) # (569, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
# print(x_train.shape) # (455, 30)
# print(x_test.shape) # (114,30)
# print(y_train.shape) # (455, 1)
# print(y_test.shape) # (114, 1)
'''
cancer_data = np.load("./data/cancer.npy")

print("cancer_data: ", cancer_data.shape)

x_train = cancer_data[:,:-1]
y_train = cancer_data[:,[-1]]


# input place holders
X = tf.placeholder(tf.float32, [None, 30])
Y = tf.placeholder(tf.float32, [None, 1]) 

print(x_train.shape, y_train.shape) # (569, 30), (569, 1)
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.2)
print(x_train.shape, y_train.shape) # (455, 30), (455, 1)

# Model
L1 = tf.layers.dense(X, 100, activation=tf.nn.leaky_relu)
L2 = tf.layers.dense(L1, 20, activation=tf.nn.leaky_relu)
L3 = tf.layers.dense(L2, 10, activation=tf.nn.leaky_relu)

logits = tf.layers.dense(L3, 1, activation=tf.nn.leaky_relu)
hypothesis = tf.nn.sigmoid(logits)

# define cost/loss & optimizer
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * 
                       tf.log(1 - hypothesis))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation 
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(3001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], 
                                        feed_dict={X: x_train, Y: y_train})
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    a, pred = sess.run([accuracy, predicted], feed_dict={X: x_test, Y: y_test})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_train.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    print(a)







'''
# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_train, Y: y_train})
        if step % 100 == 0:
            print(step, cost_val)
    
    # Accuracy report
    h, c, a = sess.run([logits, predicted, accuracy],
                        feed_dict={X: x_test, Y: y_test})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
    print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test})) 
'''