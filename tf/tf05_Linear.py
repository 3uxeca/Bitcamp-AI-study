import tensorflow as tf 
tf.set_random_seed(777)

# X and T data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = x_train * W + b # y = wx + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # compile

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost) # compile

# Launch the graph in a session.
with tf.Session() as sess: # with를 써주지 않으면 맨 밑에서 Session을 다시 close 해주어야 함
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer()) # initialize:초기화. 변수를 선언해줬으면 tf에서는 무조건 초기화를 해줘야함. 명시적으로 꼭! 쓰셈

    # Fit the line
    for step in range(2001): # keras에서의 epoch를 tf에서는 for문으로 돌린다.
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b]) # sess.run = model.fit. train이 맨앞의 _,로 간다. sess.run의 요소들이 각각 바깥에 정의된 변수로 들어감.

        if step % 20 == 0: # 20번 마다 출력
            print(step, cost_val, W_val, b_val)



# Learns best fit W:[ 1.], b:[ 0.]
'''
0 3.5240757 [2.2086694] [-0.8204183]
20 0.19749963 [1.5425726] [-1.0498911]
40 0.15214378 [1.4590572] [-1.0260718]
60 0.13793252 [1.431959] [-0.9802803]
80 0.12527035 [1.4111323] [-0.93444216]
100 0.11377248 [1.3917605] [-0.8905487]
...
1900 1.9636196e-05 [1.0051466] [-0.01169958]
1920 1.7834054e-05 [1.0049047] [-0.01114971]
1940 1.6197106e-05 [1.0046631] [-0.01060018]
1960 1.4711059e-05 [1.0044547] [-0.01012639]
1980 1.3360998e-05 [1.0042454] [-0.00965055]
2000 1.21343355e-05 [1.0040458] [-0.00919707]
'''