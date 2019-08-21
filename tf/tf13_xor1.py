import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

# X, Y, W, b, hypothesis, cost, train
# sigmoid 사용
# predict, accuracy

# placholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight') # 와꾸 잘 맞추기
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b) # sigmoid를 씌워줌으로서 결과값이 0(0.5이하)과 1(0.5이상) 사이에 수렴하도록 함

# cost/loss function 로지스틱 리그레션에서 cost에 - 가 붙는다.
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * # sigmoid에서 (-)무한대로 갈 때의 문제를 막기 위해  tf 앞에 -를 붙인다.
                       tf.log(1 - hypothesis)) # keras에서의 binary_crossentropy 를 수식처럼 풀어놓은 것이라고 생각하면 됨.

train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

# Accuracy computation 
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32)) # predicted와 Y를 비교해서 참인 값들만 다시 accuracy값 안에 넣어준다.

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
