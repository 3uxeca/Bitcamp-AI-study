import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# placholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 아래 세 줄이 한 레이어가 된다.
W1 = tf.Variable(tf.random_normal([2, 100]), name='weight') 
b1 = tf.Variable(tf.random_normal([100]), name='bias')
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1) 

W2 = tf.Variable(tf.random_normal([100, 200]), name='weight') 
b2 = tf.Variable(tf.random_normal([200]), name='bias')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2) 

W3 = tf.Variable(tf.random_normal([200, 100]), name='weight') 
b3 = tf.Variable(tf.random_normal([100]), name='bias')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3) 

W4 = tf.Variable(tf.random_normal([100, 50]), name='weight') 
b4 = tf.Variable(tf.random_normal([50]), name='bias')
layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4) 

W5 = tf.Variable(tf.random_normal([50, 80]), name='weight') 
b5 = tf.Variable(tf.random_normal([80]), name='bias')
layer5 = tf.sigmoid(tf.matmul(layer4, W5) + b5) 

W6 = tf.Variable(tf.random_normal([80, 120]), name='weight') 
b6 = tf.Variable(tf.random_normal([120]), name='bias')
layer6 = tf.sigmoid(tf.matmul(layer5, W6) + b6) 

W7 = tf.Variable(tf.random_normal([120, 100]), name='weight') 
b7 = tf.Variable(tf.random_normal([100]), name='bias')
layer7 = tf.sigmoid(tf.matmul(layer6, W7) + b7) 

W8 = tf.Variable(tf.random_normal([100, 180]), name='weight') 
b8 = tf.Variable(tf.random_normal([180]), name='bias')
layer8 = tf.sigmoid(tf.matmul(layer7, W8) + b8) 

W9 = tf.Variable(tf.random_normal([180, 100]), name='weight') 
b9 = tf.Variable(tf.random_normal([100]), name='bias')
layer9 = tf.sigmoid(tf.matmul(layer8, W9) + b9) 

W10 = tf.Variable(tf.random_normal([100, 1]), name='weight') 
b10 = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.sigmoid(tf.matmul(layer9, W10) + b10) 
 
# cost/loss function 로지스틱 리그레션에서 cost에 - 가 붙는다.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * 
                       tf.log(1 - hypothesis)) 

train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

# Accuracy computation 
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32)) 

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(8001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", c, "\nCorrect (Y): ", y_data, "\nAccuracy: ", a)
