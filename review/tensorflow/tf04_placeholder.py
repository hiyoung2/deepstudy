import tensorflow as tf


# 1. constant
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

sess = tf.Session()

# 2. placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

# feed_dict : placeholder에 값을 집어 넣어주게 함!
# run 하는 과정에서 사용한다
# 1처럼 placeholder에 상수 하나가 들어가기도 하고
# 2처럼 placeholder에 리스트 형태로 상수 두 개 이상이 들어가기도 한다 -> 행렬 형태도 가능한가,,
print(sess.run(adder_node, feed_dict = {a:3, b:4.5})) # 1
print(sess.run(adder_node, feed_dict = {a:[1, 3], b:[2, 4]})) # 2


add_and_triple = adder_node * 3
print(sess.run(add_and_triple, feed_dict = {a:3, b:4.5}))
# print(Session명.run(결괏값명, feed_dict = {인풋되는 값(들)}))
