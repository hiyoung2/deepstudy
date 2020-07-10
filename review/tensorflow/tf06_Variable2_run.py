# Variable1_run 파일에서 배운 3가지 방법을 모두 써보면서 hypothesis 구하기 실습

import tensorflow as tf

tf.set_random_seed(777)

x = [1.0, 2.0, 3.0]
w = tf.Variable([0.3])
b = tf.Variable([1.0])

hypothesis = w*x + b

print("x :", x)
print("w :", w)
print("b :", b)

# x : [1.0, 2.0, 3.0]
# w : <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>
# b : <tf.Variable 'Variable_1:0' shape=(1,) dtype=float32_ref>

# 1) 정석?
sess = tf.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypothesis)
print("hypothesis :", aaa)
sess.close()

print("------------------------------------------------------")

# 2) eval 사용
sess = tf.Session()
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval(session = sess)
print("hypothesis :", bbb)
sess.close()

print("------------------------------------------------------")

# 3) InteractiveSession과 eval 사용
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval()
print("hypothesis :", ccc)
sess.close()

print("------------------------------------------------------")
