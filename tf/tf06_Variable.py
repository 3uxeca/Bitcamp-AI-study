# 랜덤값으로 변수 1개를 만들고, 변수의 내용을 출력하시오.

import tensorflow as tf
# tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")


# print(W)
W = tf.Variable([0.3], tf.float32) # W = 0.3

with tf.Session() as sess: # with를 써주지 않으면 맨 밑에서 Session을 다시 close 해주어야 함
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer()) # initialize:초기화. 변수를 선언해줬으면 tf에서는 무조건 초기화를 해줘야함. 명시적으로 꼭! 쓰셈
    print(sess.run(W))
    print(sess.run(b))


# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# # print(sess.run(W))
# print(sess.run(b))
# sess.close()

# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# aaa = W.eval()
# print(aaa)
# sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = W.eval(session=sess)
print(aaa)
sess.close()