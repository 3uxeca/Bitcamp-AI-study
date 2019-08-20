import tensorflow as tf 
import numpy as np
tf.set_random_seed(777)

sess = tf.Session()

xy = np.loadtxt('./data/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1] 
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape) # x.shape(101, 16) / y.shape(101,1)

from keras.utils import np_utils
y_data = np_utils.to_categorical(y_data)

print(x_data.shape, y_data.shape) # x.shape(101, 16) / y.shape(101,7)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 16])
Y = tf.placeholder(tf.float32, shape=[None, 7])
nb_classes = 7

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')


# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) # categorical_crossentropy 수식

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Accuracy computation 
# True if hypothesis>0.5 else False
predicted = tf.argmax(hypothesis, 1)
# predicted_ = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
predicted_ = tf.equal(predicted, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(predicted_, dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, p, c, a = sess.run([hypothesis, predicted, predicted_, accuracy],
                        feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nPredicted : ", p, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
