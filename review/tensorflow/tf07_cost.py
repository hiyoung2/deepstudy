# 시각화

import tensorflow as tf
import matplotlib.pyplot as plt

x = [1., 2., 3.]
y = [1., 2., 3.]

w = tf.placeholder(tf.float32)

hypothesis = x * w

# 평균 제곱 오차를 loss로 사용(손실 함수 : 평균 제곱 오차)
cost = tf.reduce_mean(tf.square(hypothesis - y))

# 갱신 될 w, cost 즉, 가중치와 loss를 담을 공간 준비
w_history = []
cost_history = []

with tf.Session() as sess :
    for i in range(-30, 50) :
        curr_w = i * 0.1 # 가중치를 0, 0.1, 0.2 , ...로 설정
        curr_cost = sess.run(cost, feed_dict={w : curr_w}) # placeholder로 준비한 w에 feed_dict로 값을 넣어준다
        # curr_cost를 세션을 통과시켜 실행하여 값을 볼 수 있도록 만든다

        w_history.append(curr_w) # for문을 돌면서 갱신되는 w들을 빈 리스트에 append
        cost_history.append(curr_cost) # cost(loss) 역시 갱신되므로 위와 같이 빈 리스트에 append

plt.plot(w_history, cost_history)
plt.show()