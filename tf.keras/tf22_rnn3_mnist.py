import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

## 옵션 설정
learning_rate = 0.001
total_epoch = 30
batch_size = 128

# 가로 픽셀수를 n_input 으로, 세로 픽셀수를 입력 단계인 n_step으로 설정
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

#######
# 신경망 모델 구성
#######
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# print(X) # (?, 28, 28)
# print(Y) # (?, 10)
# print(W) # (128, 10)
# print(b) # (10, )

# RNN 학습에 사용할 셀 생성
# 다음 함수들을 사용하면 다른 구조의 셀로 간단하게 변경할 수 있다.
# BasicRNNCell, BasicLSTMCell, GRUCell
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden) # n_hidden => 첫번째 아웃풋 ########### 중요중요중요중요중요 와꾸중요

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) # X => 첫번째 레이어 shape ########  X가 통째로 들어가는 것 주의

# print(outputs) # (?, 28, 128)

# 결과를 Y의 다음 형식과 바꿔야 하기 때문에
# Y : [batch_size, n_class]
# outputs 의 형태를 이에 맞춰 변경
# outputs : [batch_size, n_step, n_hidden]
#        -> [n_step, batch_size, n_hidden]
#        -> [batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])
# print(outputs) # (28, ?, 128)
outputs = outputs[-1]
# print(outputs) # (?, 128)

model = tf.matmul(outputs, W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

###################
# 신경망 모델 학습 #
###################
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # X 데이터를 RNN 입력 데이터에 맞게 [batch_size, n_step, n_input] 형태로 변환
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost],
                                feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch: ', '%04d' % (epoch + 1),
          'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

#############
# 결과 확인  #
#############
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('정확도: ', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))
