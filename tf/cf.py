import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split
tf.set_random_seed(77)

cancer_data = np.load("./data/cancer.npy")

print("cancer_data:",cancer_data.shape) # (569, 31)

x_train = cancer_data[:,:-1]

y_train = cancer_data[:,[-1]]


X = tf.placeholder(tf.float32,[None, 30])
Y = tf.placeholder(tf.float32,[None, 1])

# print(x_train.shape, y_train.shape) # (569, 30), (569, 1)
x_train, x_test, y_train, y_test = train_test_split(x_train,y_train,test_size=0.2)
# print(x_train.shape, y_train.shape) # (455, 30), (455, 1)

# w1 = tf.get_variable("w1",shape=[?,?],initializer=tf.random_uniform_initializer())
# b1 = tf.Variable(tf.random_normal([512]))


# tf.constant_initializer()
# tf.zeros_initializer()
# tf.random_uniform_initializer()
# tf.random_normal_initializer()
# tf.contrib.layers.xavier_initializer()

l1 = tf.layers.dense(X, 100, activation=tf.nn.leaky_relu)
l2 = tf.layers.dense(l1, 20, activation=tf.nn.leaky_relu)
l3 = tf.layers.dense(l2, 10, activation=tf.nn.leaky_relu)
logits = tf.layers.dense(l3, 1,activation=tf.nn.leaky_relu)






hypothesis = tf.nn.sigmoid(logits)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))



train = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))    # 일반적인 선형 회귀에선 안된다




with tf.Session() as sess:
    # Initialize TensorFlow variables
    

    sess.run(tf.global_variables_initializer())
    
    for step in range(3000):
        _, cost_val, acc_val = sess.run([train, cost, accuracy], feed_dict={X: x_train, Y: y_train})
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:f}\tAcc: {:.2%}".format(step, cost_val, acc_val))
            
    # Let's see if we can predict
    a, pred = sess.run([accuracy, predicted], feed_dict={X: x_test,Y: y_test})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_train.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
    print(a)
    writer = tf.summary.FileWriter('./board/sample_1', sess.graph)
