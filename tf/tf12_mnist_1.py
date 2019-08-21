# Batch & Epoch and shape(784)

import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print(mnist.train.images)
# print(mnist.test.labels)
# print(mnist.train.images.shape) # (55000, 784)
# print(mnist.test.labels.shape) # (10000, 10)
# print(type(mnist.train.images)) # <class 'numpy.ndarray'>

###코딩하시오  X, Y, W, b, hypothesis, cost, train

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 784]) # 784 = 28 * 28 * 1
Y = tf.placeholder(tf.float32, shape=[None, 10])
nb_classes = 10

W = tf.Variable(tf.random_normal([784, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1)) # categorical_crossentropy 수식

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost) # 0.8951
# train = tf.train.AdamOptimizer(learning_rate=0.00015).minimize(cost) # 0.7873
# train = tf.train.AdadeltaOptimizer(learning_rate=0.9).minimize(cost) # 0.8592
# train = tf.train.AdagradOptimizer(learning_rate=0.008).minimize(cost) # 0.7636

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations): # 전체 데이터를 batch_size로 나누는 것! 55000 / 100 = 550
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # 100개를 뽑아서 550번을 돌겠다
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}
        ),
    )

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1]}),
    )

    plt.imshow(
        mnist.test.images[r : r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()