# Minimizing Cost
import tensorflow as tf 
import matplotlib.pyplot as plt

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Variables for plotting cost function
W_history = []
cost_history = []

# Launch the graph in a session.
with tf.Session() as sess:
    for i in range(-30, 50):
        curr_W = i * 0.1 # 0.1 간격으로 표시되게끔 함
        curr_cost = sess.run(cost, feed_dict={W: curr_W})

        W_history.append(curr_W) # 위에서 만든 리스트의 뒷쪽에 추가
        cost_history.append(curr_cost)

# Show the cost function
plt.plot(W_history, cost_history)
plt.show()