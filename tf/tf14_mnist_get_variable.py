import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

keep_prob = tf.placeholder(tf.float32)

# W1 = tf.get_variable("W1", shape=[?, ?],
#                      initializer=tf.random_uniform_initializer())
# b1 = tf.Variable(tf.random_normal([512]))
# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# tf.constant_initializer()
# tf.zeros_initializer() # 0.9655
# tf.random_uniform_initializer()
# tf.random_normal_initializer()
# tf.contrib.layers.xavier_initializer() # 0.9738 # 얘가 제일 좋음. get_variable 할때 무조건 초기화 옵션 써줘야함

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 784]) # 784 = 28 * 28 * 1
Y = tf.placeholder(tf.float32, shape=[None, 10])
nb_classes = 10

W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer()) 
b1 = tf.Variable(tf.random_normal([256]), )
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1) 
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob) 

W2 = tf.get_variable("W2", shape=[256, 256], initializer=tf.contrib.layers.xavier_initializer()) 
b2 = tf.Variable(tf.random_normal([256]),)
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2) 
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[256, nb_classes], initializer=tf.contrib.layers.xavier_initializer()) 
b3 = tf.Variable(tf.random_normal([nb_classes]), )
hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3) 

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(hypothesis, Y))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations): 
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) 
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7})
            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print(
        "Accuracy: ",
        accuracy.eval(
            session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}
        ),
    )

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print(
        "Prediction: ",
        sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r : r + 1], keep_prob: 1}),
    )

    plt.imshow(
        mnist.test.images[r : r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()