# mv : multi variable

import tensorflow as tf
tf.set_random_seed(777)

# input : 3, output : 1
# 1. 데이터 준비
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

# 위의 데이터들을 x1, x2, x3 이라는 placeholder에 넣어준다(sess.run 실행 시, feed_dict를 통해)
x1 = tf.placeholder(tf.float32) # shape를 안 써줘도 된다? -> 행렬 형태가 아닌 scalar들을 모아 놓은 단순 vector라서 shape 입력을 따로 안 해준다?
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

y = tf.placeholder(tf.float32)

# 입력 데이터의 수에 맞게끔 weight를 준비
w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 2. 모델 구성
# keras에서 multi perceptron? 다중 퍼셉트론과 같은 것?
# 각각 입력 데이터들에 가중치를 곱해준 것을 단순 더하기로 병합?
# keras의 모델과 비교가 필요

hypothesis = (x1 * w1) + (x1 * w2) + (x3 * w3) + b
# matmul을 사용하여 이렇게 써도 된다
# hypothesis = tf.matmul([(x1, w1), (x2, w2), (x3, w3)]) + b

cost = tf.reduce_mean(tf.square(hypothesis - y))

lr = 1e-4
# train 쓰는 방법 2가지

# 1. optimizer와 손실함수 최소화 방법을 한 번에 모두 써 주는 방법
train = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(cost)

# 2. optimizer와 손실함수 최소화 방법을 분리해서 써 주는 방법
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
# train = optimizer.minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {x1:x1_data, x2:x2_data, x3:x3_data, y:y_data})

        if step % 100 == 0 :
            print(step, "cost :", cost_val, "\n", hy_val)

# nan 뜨니까 수정 필요!