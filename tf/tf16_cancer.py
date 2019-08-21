# cancer.npy를 가지고 tensorflow 코딩을 하시오
# test와 train 분리할 것
# dropout, get_variable, multi layer 등 배운 것을 모두 사용할 것
import tensorflow as tf
import numpy as np
import random

tf.set_random_seed(777)

x_data = np.load("./data/cancer_x.npy", allow_pickle = True)
y_data = np.load("./data/cancer_y.npy", allow_pickle = True)
# print(x_data.shape) # (569, 30)
# print(y_data.shape) # (569, )
# print(x_data)

y_data = y_data.reshape(y_data.shape[0], 1)
# print(y_data.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
# print(x_data[:10])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
                                    x_data, y_data, test_size=0.2)

# print(x_train.shape) # (455, 30)
# print(x_test.shape) # (114, 30)
# print(y_train.shape) # (455, 1)
# print(y_test.shape) # (114, 1)
# print(y_train[:10])

keep_prob = tf.placeholder(tf.float32)

# nb_classes = 2 # 0,1

X = tf.placeholder(tf.float32, shape=[None, 30])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Y_one_hot = tf.one_hot(Y, nb_classes)
# print("one_hot:", Y_one_hot)
# Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
# print("reshape one_hot:", Y_one_hot)

'''
# one_hot: Tensor("one_hot:0", shape=(?, 3), dtype=float32)
# reshape one_hot: Tensor("Reshape:0", shape=(?, 3), dtype=float32)
'''

# W1 = tf.get_variable("W1", shape=[30, 50], initializer=tf.contrib.layers.xavier_initializer()) 
# b1 = tf.Variable(tf.random_normal([50]))
# layer1 = tf.nn.relu(tf.matmul(X, W1) + b1) 
# layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob) 

# W2 = tf.get_variable("W2", shape=[50, 50], initializer=tf.contrib.layers.xavier_initializer()) 
# b2 = tf.Variable(tf.random_normal([50]))
# layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2) 
# layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[30, 1], initializer=tf.contrib.layers.xavier_initializer()) 
b3 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.sigmoid(tf.matmul(X, W3) + b3) 

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * 
                       tf.log(1 - hypothesis))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_test, 1))
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
'''
sess = tf.Session()

pred = sess.run([hypothesis], feed_dict={X: x_test, keep_prob: 1})
pred = np.array(pred) # 원본데이터와 형식 통일

y_test = y_test.reshape((-1,)) # flatten
pred = pred.reshape((-1,))


# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_data, pred): # y_test, y_predict의 차이를 비교하기 위한 함수
    return np.sqrt(mean_squared_error(y_test, pred)) # np.sqrt 제곱근씌우기
print("RMSE : ", RMSE(y_test, pred))

# R2(결정계수) 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, pred)
print("R2 : ", r2_y_predict)

sess.close()
'''