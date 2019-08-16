import tensorflow as tf

node1 = tf.constant(3.0, tf.float32) # constant : 상수. 변하지 않는 값. node1에 tf의 상수형태로 3.0을 tf.float32(실수형)형식으로 넣겠다.
node2 = tf.constant(4.0) # tf.float32 가 default 값
node3 = tf.add(node1, node2) # node1과 node2를 더함

# print("node1:", node1, "node2", node2)
# print("node3:", node3)

sess = tf.Session()
# print("sess.run(node1, node2): ", sess.run([node1, node2]))
# print("sess.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # _ provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5})) # 3 + 4.5 = 7.5
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]})) # [1, 3] + [2, 4] = [3, 7]

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict={a: 3, b:4.5})) # (3 + 4.5) * 3 = 22.5