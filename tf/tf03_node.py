import tensorflow as tf

node1 = tf.constant(3.0, tf.float32) # constant : 상수. 변하지 않는 값. node1에 tf의 상수형태로 3.0을 tf.float32(실수형)형식으로 넣겠다.
node2 = tf.constant(4.0) # tf.float32 가 default 값
node3 = tf.add(node1, node2) # node1과 node2를 더함

print("node1:", node1, "node2", node2)
print("node3:", node3)

'''
= print 결과 =
node1: Tensor("Const:0", shape=(), dtype=float32) node2 Tensor("Const_1:0", shape=(), dtype=float32)
              첫번째상수,  모른다, 실수형이다.
node3: Tensor("Add:0", shape=(), dtype=float32)
              첫번째상수, 모른다, 실수형이다. ===> Graph의 '형태'만 출력된다.
'''

sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))

'''
sess.run(node1, node2):  [3.0, 4.0]
sess.run(node3):  7.0
'''