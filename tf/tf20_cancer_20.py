import numpy as np 
import tensorflow as tf
import random
tf.set_random_seed(777)

def next_batch(num, data, labels):
    
#   `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.

    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)



# hyper parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 50

# Iris Data
x_data = np.load("./data/cancer_x.npy")
y_data = np.load("./data/cancer_y.npy")

# print(x_data.shape) # (569, 30)
# print(y_data.shape) # (569, )

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, shuffle=True)

y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
# print(x_train.shape) # (455, 30)
# print(x_test.shape) # (114,30)
# print(y_train.shape) # (455, 1)
# print(y_test.shape) # (114, 1)

# input place holders
X = tf.placeholder(tf.float32, [None, 30])
Y = tf.placeholder(tf.float32, [None, 1]) 

# Model
L1 = tf.layers.dense(X, 128, activation=tf.nn.relu)
# L2 = tf.layers.dense(L1, 128, activation=tf.nn.relu)
# L3 = tf.layers.dense(L2, 84, activation=tf.nn.relu)
L2 = tf.layers.dense(L1, 64, activation=tf.nn.relu)
L3 = tf.layers.dense(L2, 32, activation=tf.nn.relu)

logits = tf.layers.dense(L3, 1, activation=tf.sigmoid)

# define cost/loss & optimizer
cost = -tf.reduce_mean(Y * tf.log(logits) + (1 - Y) * 
                       tf.log(1 - logits))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation 
# True if hypothesis>0.5 else False
predicted = tf.cast(logits > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        _, cost_val, acc_val = sess.run([optimizer, cost, accuracy], 
                                        feed_dict={X: x_train, Y: y_train})
        if step % 100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if we can predict
    pred = sess.run(predicted, feed_dict={X: x_test})
    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_test.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

    print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test})) 




# # Launch graph
# with tf.Session() as sess:
#     # Initialize TensorFlow variables
#     sess.run(tf.global_variables_initializer())

#     for step in range(1001):
#         cost_val, _ = sess.run([cost, optimizer], feed_dict={X: x_train, Y: y_train})
#         if step % 100 == 0:
#             print(step, cost_val)
    
#     # Accuracy report
#     h, c, a = sess.run([logits, predicted, accuracy],
#                         feed_dict={X: x_test, Y: y_test})
#     print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
#     print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test})) 
