# iris.npy를 가지고 tensorflow 코딩을 하시오
# test와 train 분리할 것
# dropout, get_variable, multi layer 등 배운 것을 모두 사용할 것
import tensorflow as tf
import numpy as np
import random

tf.set_random_seed(777)

iris = np.load("./data/iris.npy", allow_pickle = True)
# print(iris.shape) # (150, 5)
# print(iris)

x_data = iris[:, 0:-1]
y_data = iris[:, [-1]]
# print(x_data.shape, y_data.shape) # (150, 4), (150, 1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
                                    x_data, y_data, test_size=0.2)

# print(x_train.shape) # (120, 4)
# print(x_test.shape) # (30, 4)
# print(y_train.shape) # (120, 1)
# print(y_test.shape) # (30, 1)

keep_prob = tf.placeholder(tf.float32)

nb_classes = 3 # 1,2,3

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:", Y_one_hot)

'''
one_hot: Tensor("one_hot:0", shape=(?, 1, 3), dtype=float32)
reshape one_hot: Tensor("Reshape:0", shape=(?, 3), dtype=float32)
'''

W1 = tf.get_variable("W1", shape=[4, 50], initializer=tf.contrib.layers.xavier_initializer()) 
b1 = tf.Variable(tf.random_normal([50]), )
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1) 
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob) 

W2 = tf.get_variable("W2", shape=[50, 50], initializer=tf.contrib.layers.xavier_initializer()) 
b2 = tf.Variable(tf.random_normal([50]),)
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2) 
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[50, nb_classes], initializer=tf.contrib.layers.xavier_initializer()) 
b3 = tf.Variable(tf.random_normal([nb_classes]), )
hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3) 

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(hypothesis, Y_one_hot))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
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
    for p, y in zip(pred, y_test.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob: 1}))