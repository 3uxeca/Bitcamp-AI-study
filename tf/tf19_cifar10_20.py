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
training_epochs = 100
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

L1 = tf.layers.conv2d(X, 120, [3,3], activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(L1, [2,2], [2,2])
L1 = tf.layers.dropout(L1, 0.7)
# print("L1", L1) # L1 Tensor("dropout/Identity:0", shape=(?, 15, 15, 120), dtype=float32)


L2 = tf.layers.conv2d(L1, 120, [3,3], activation=tf.nn.relu)
L2 = tf.layers.max_pooling2d(L2, [2,2], [2,2])
L2 = tf.layers.dropout(L2, 0.5)
# print("L2", L2) # L2 Tensor("dropout_1/Identity:0", shape=(?, 6, 6, 70), dtype=float32)

L3 = tf.layers.conv2d(L2, 80, [3,3], activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.25)

L4 = tf.layers.conv2d(L3, 60, [3,3], activation=tf.nn.relu)

L5 = tf.contrib.layers.flatten(L4)
L5 = tf.layers.dense(L5, 128, activation=tf.nn.relu)

logits = tf.layers.dense(L5, 10, activation=None)

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
