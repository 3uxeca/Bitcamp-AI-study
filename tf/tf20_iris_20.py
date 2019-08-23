import numpy as np 
import tensorflow as tf
import random
tf.set_random_seed(777)

# hyper parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100

# Iris Data
x_data = np.load("./data/iris_x.npy")
y_data = np.load("./data/iris_y.npy")

# print(x_data.shape) # (150, 4)
# print(y_data.shape) # (150, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, shuffle=True)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
# print(x_train.shape) # (120, 4)
# print(x_test.shape) # (30, 4)
# print(y_train.shape) # (120, 1)
# print(y_test.shape) # (30, 1)

# input place holders
X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.int32, [None, 1]) 

nb_classes = 3

Y_one_hot = tf.one_hot(Y, nb_classes) # one-hot
# print("one_hot:", Y_one_hot) # one_hot: Tensor("one_hot:0", shape=(?, 1, 3), dtype=float32)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# print("reshape one_hot:", Y_one_hot) # reshape one_hot: Tensor("Reshape:0", shape=(?, 3), dtype=float32)

# Model
L1 = tf.layers.dense(X, 128, activation=tf.nn.relu)
L2 = tf.layers.dense(L1, 128, activation=tf.nn.relu)
L3 = tf.layers.dense(L2, 84, activation=tf.nn.relu)
L4 = tf.layers.dense(L3, 64, activation=tf.nn.relu)
L5 = tf.layers.dense(L4, 32, activation=tf.nn.relu)

logits = tf.layers.dense(L5, 3, activation=tf.nn.softmax)

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                        labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], 
                                        feed_dict={X: x_train, Y: y_train})
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_test})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_test.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

    print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test})) 
