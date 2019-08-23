# keras dataset cifar10을 tf 14버전으로 코딩하되, 출력은 acc로

from keras.datasets import cifar10
import numpy as np 
import tensorflow as tf 
import random
import matplotlib.pyplot as plt

def next_batch(num, data, labels):
    
#   `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


tf.set_random_seed(777)

# hyper parameters
learning_rate = 0.001
training_epochs = 50
batch_size = 100

# CIFAR10 Data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# # print(X_train.shape[0], 'train samples') # 50000
# # print(X_train.shape, 'train samples') # (50000, 32, 32, 3)
# # print(X_test.shape[0], 'test samples') # 10000
# # print(X_test.shape, 'test samples') # (10000, 32, 32, 3)
# print(y_train.shape[0], 'train samples') # 50000
# print(y_train.shape, 'train samples') # (50000, 1)
# print(y_test.shape[0], 'test samples') # 10000
# print(y_test.shape, 'test samples') # (10000, 1)


# input place holders
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
# X_img = tf.reshape(X, [-1, 32, 32, 3])       # img 32x32x3 (color img)
Y = tf.placeholder(tf.int32, [None, 1]) 

nb_classes = 10

Y_one_hot = tf.one_hot(Y, nb_classes) # one-hot
print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot:", Y_one_hot)

# one_hot: Tensor("one_hot:0", shape=(?, 1, 10), dtype=float32)
# reshape one_hot: Tensor("Reshape:0", shape=(?, 10), dtype=float32)


# L1 ImgIn shape=(?, 32, 32, 3)
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01)) # W값에 kernal_size=(3,3)으로 자를거고, 컬러이미지이고, 마지막값은 output값
# print('W1 : ', W1) # W1 :  <tf.Variable 'Variable:0' shape=(3, 3, 3, 32) dtype=float32_ref>

#   conv    -> (?, 32, 32, 32)
#   Pool    -> (?, 16, 16, 32)
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')  # strides 몇칸씩 움직일것인가? [0] [3]의 1은 의미없는 디폴트 값. 한칸씩 움직인다.
# print('L1 : ', L1) # L1 :  Tensor("Conv2D:0", shape=(?, 32, 32, 32), dtype=float32)

L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], # ksize도 strides와 동일하게 양끝의 1은 의미가 없다. 반으로 줄이겠다
                    strides=[1, 2, 2, 1], padding='SAME') # padding이 없을 경우  # 전체갯수 - 몇개로 자를 것인지 + 1 

# print('L1 : ', L1) # L1 :  Tensor("MaxPool:0", shape=(?, 16, 16, 32), dtype=float32)

# L2 ImgIn shape=(?, 16, 16, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # 3x3, W1의 output->input output 64
#   conv    -> (?, 16, 16, 64)
#   Pool    -> (?, 8, 8, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME') # L1을 3x3으로 자르겠다. 한칸씩 동일한 패딩
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
# print('L2 : ', L2) # L2 :  Tensor("MaxPool_1:0", shape=(?, 8, 8, 64), dtype=float32)

# L3 ImgIn shape=(?, 8, 8, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=0.01))
#   conv    -> (?, 8, 8, 32)
#   Pool    -> (?, 4, 4, 32)                    
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
# L3 = tf.nn.dropout(L3, 0.7)
# print('L3 : ', L3) # L3 :  Tensor("MaxPool_2:0", shape=(?, 4, 4, 32), dtype=float32)
L3_flat = tf.reshape(L3, [-1, 4 * 4 * 32])


W4 = tf.get_variable("W4", shape=[4 * 4 * 32, 50], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([50]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b1)

W5 = tf.get_variable("W5", shape=[50, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([10]))
logits = tf.nn.softmax(tf.matmul(L4, W5) + b2)


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient([Y_one_hot])))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model # model.fit 부분을 for문으로 수행
print('Learning started. It takes sometime.')
for epoch in range(training_epochs): # 전체 돌아가는 횟수 epochs
    avg_cost = 0
    total_batch = X_train.shape[0] // batch_size

    for i in range(total_batch): # 15 epochs를 100개로 잘라서 수행 => batch_size
        batch_xs, batch_ys = next_batch(batch_size, X_train, y_train)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
pre_ = tf.argmax(logits, 1)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.arg_max(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: X_test, Y: y_test})) 


# Get one and predict
r = random.randint(0, (len(y_test)) - 1)
print("Label: ", sess.run(tf.argmax(y_test[r:r + 1], 1)))
print("Prediction: ", sess.run( tf.argmax(logits, 1),
                                feed_dict={X: X_test[r:r + 1]}))    

