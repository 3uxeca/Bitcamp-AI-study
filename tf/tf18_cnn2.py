import tensorflow as tf
import random
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # one hot이 되서 나온다.

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])       # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01)) # W값에 kernal_size=(3,3)으로 자를거고, 흑백이미지이고, 마지막값은 output값
print('W1 : ', W1) # W1 :  <tf.Variable 'Variable:0' shape=(3, 3, 1, 32) dtype=float32_ref> 와꾸 나왔다.

#   conv    -> (?, 28, 28, 32)
#   Pool    -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # strides 몇칸씩 움직일것인가? [0] [3]의 1은 의미없는 디폴트 값. 한칸씩 움직인다.
print('L1 : ', L1) # L1 :  Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)

L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], # ksize도 strides와 동일하게 양끝의 1은 의미가 없다. 반으로 줄이겠다
                    strides=[1, 2, 2, 1], padding='SAME') # padding이 없을 경우  # 전체갯수 - 몇개로 자를 것인지 + 1 

print('L1 : ', L1) # L1 :  Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # 3x3, W1의 output->input output 64
#   conv    -> (?, 14, 14, 64)
#   Pool    -> (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') # L1을 3x3으로 자르겠다. 한칸씩 동일한 패딩
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7 * 7 * 64]) # -1: 전체 행. flatten은 전부다 곱하는거!

##### 8/23 시작 ####
# Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[7 * 7 * 64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                      logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model # model.fit 부분을 for문으로 수행
print('Learning started. It takes sometime.')
for epoch in range(training_epochs): # 전체 돌아가는 횟수 epochs
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch): # 15 epochs를 100개로 잘라서 수행 => batch_size
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run( tf.argmax(logits, 1),
                                feed_dict={X: mnist.test.images[r:r + 1]}))       