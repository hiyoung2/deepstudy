import tensorflow as tf

tf.set_random_seed(777) # 난수 설정

# y = 2x + 1 이 되는 (아주 깔끔하게 정제된) 데이터
x_train = [1, 2, 3]
y_train = [3, 5, 7]

# KEYWORD : Variable
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 많이 보고 들어왔던 것들이다
# 가중치, 편향
# y = wx + b

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train)) 
# 예측값 - 실젯값을 뺀 것을 제곱하여 평균? -> mse! : 평균 제곱 오차


train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
# train 시킬 것인데
# GradientDescentOptimizer 라는 옵티마이저를 사용하겠다! == SGD (다른 옵티마이저도 사용 가능할 것)
# 옵티마이저에 학습률 적용
# cost, 즉 mse를 최소화하는 방식으로!

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # 설정된 전체 변수들이 모두 초기화 된다, global이 전역임을 설정해주므로 모두 초기화 되는 것, 이 파일 내에서

    for step in range(2001) :
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b]) # keras로 따지면 compile 과정이라 볼 수 있다
        # _ : 이렇게 쓰면 출력하지 않겠다는 의미, 즉 여기서는 train 의 결과는 따로 출력하지 않겠다는 의미가 됨
        
        if step % 20 == 0 :
            print(step, cost_val, W_val, b_val) # 총 epoch는 2000번인데, 20번마다 보여주도록 if문 적용