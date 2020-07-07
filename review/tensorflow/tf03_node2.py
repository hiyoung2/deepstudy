# 덧셈을 해 보았으니 사칙연산 도전
# 3 + 4 + 5
# 4 - 3
# 3 * 4
# 4 / 2

import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(5.0)
node4 = tf.constant(2.0)

# constant 형태로 넣은 숫자들을 Session을 거쳐 우리가 원하는 딱 '숫자'의 형태로 만들어줘야 함
# 연산작업을 위한 기본적인 처리!
sess = tf.Session()
print(sess.run([node1, node2, node3, node4])) 
# [3.0, 4.0, 5.0, 2.0]

node_add = tf.add_n([node1, node2, node3]) # 앞서 알던 형태와 다른 덧셈 장치
                                           # 3개 이상의 요소들을 연산을 해야 할 때 '_n'을 덧붙여줘야 한다
                                           # 많은 양의 텐서들을 한 번에 처리할 때 사용하는 것

node_sub = tf.subtract(node2, node1)
node_mul = tf.multiply(node1, node2)
node_div = tf.divide(node2, node4)

print(node_add) # 이렇게 하면 우리가 원하는 사칙연산 결과가 아니라 자료형이 나올 것
# Tensor("AddN:0", shape=(), dtype=float32)

print(sess.run([node_add, node_sub, node_mul, node_div]))
# [12.0, 1.0, 12.0, 2.0]