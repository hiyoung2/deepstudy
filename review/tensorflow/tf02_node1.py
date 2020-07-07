# visual studio code 하단, path 설정에서 'tf114' 확인!
# 'tf114'라고 만든 가상환경에서만 tensorflow 1.14 version 작동

import tensorflow as tf
node1 = tf.constant(3.0, tf.float32) # 테스트 해 봤는데 tf.float32 지정해주지 않아도 실수형으로 나오긴 함
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2) # 덧셈을 해 보자!

# Session 을 거치지 않았으므로 자료형 형식으로 출력될 것
print("node1 :", node1, "node2 :", node2)
print("node3 :", node3)

'''
node1 : Tensor("Const:0", shape=(), dtype=float32) node2 : Tensor("Const_1:0", shape=(), dtype=float32)
node3 : Tensor("Add:0", shape=(), dtype=float32)
'''

# 우리가 원하는 형태, 즉 연산 결과로 출력해보자
sess = tf.Session()
print("sess.run(node1, node2 :", sess.run([node1, node2])) # 리스트를 사용하여 두 노드를 출력
print("sess.run(node3) :", sess.run(node3))
'''
sess.run(node1, node2 : [3.0, 4.0]
sess.run(node3) : 7.0
'''