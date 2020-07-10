##################################################################
# tf.constant vs tf.Variable vs tf.placeholder
##################################################################

# tf.constant : 텐서플로우, 변하지 않는 상수 생성
# - 잘 사용되지는 않음
# tf.Variable : 값이 바뀔 수도 있는 변수 생성
# - 단, 변수는 그래프를 실행하기 전에 초기화를 해줘야 한다
# - Session을 초기화(tf.global_variables_initializer())하는 순간, 변수에 그 값이 지정된다

# tf.placeholder : 일정 값을 받을 수 있게 만들어주는 그릇을 생성한다
# placeholder를 이용할 때에는, 실행 시(run) feed_dict = {x : '입력값'}와 같은 식으로 값을 지정한다



import tensorflow as tf
tf.set_random_seed(777) # 난수 생성, tf.random_normal에 영향?

# 1. 데이터(사실상, 입력값은 넣지 않은 상태)
# 모델에 사용할 입력 데이터와 출력 데이터를 placeholder로 받음
# placeholder는 feed_dict를 통해 값을 입력하는데
# feed_dict는 sess.run(), 즉 Session을 통과할 때에 써 준다
x_train = tf.placeholder(tf.float32, shape = [None])
y_train = tf.placeholder(tf.float32, shape = [None])

# weight와 bias는 Variable로 생성
w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 2. 모델 구성
# 선형 모델을 정의(입력 데이터와 가중치 w, 편향 b를 사용하여)
hypothesis = x_train * w + b

# 3. 컴파일, 훈련
# 여기서 쓰인 cost는 케라스로 따지면 loss와 같다
# 손실 함수를 정의
cost = tf.reduce_mean(tf.square(hypothesis -y_train))

# learning_rate를 lr이라는 변수를 통해 값을 미리 지정
lr = 1e-4

# 평균 제곱 오차가 최소화 되는 지점을 경사하강법으로 구한다
# 경사하강법으로 손실 함수를 최소화
train = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(cost)


# 4. 평가, 예측
# tensor의 특징 : tf.Session()을 통과해야 한다
# Session 시작
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) # 전역(이 파일 모든 범위)의 변수를 초기화 
                                                # 여기서 초기화란, '0'으로 만드는 것이 아니라, 변수를 선언한다는 의미가 된다

    for step in range(2001) : # 여기서 step은 keras로 따지면 epoch가 된다(range(2001)은 2000번!)
        _, cost_val, w_val, b_val = sess.run([train, cost, w, b], feed_dict = {x_train : [1, 2, 3], y_train : [3, 5, 7]})
        # _, cost_val, w_val, b_val : 출력시킬 때 사용할 변수명(_ : 따로 출력하지 않음)
        # 위의 것들을 Session을 통과하여 실행시킴 , 각가의 자리에 train, cost, w, b가 들어감(리스트로 묶어서 입력)
        # placeholder로 지정만 하고 값을 넣어주지 않았는데, sess.run 하는 여기에서 feed_dict를 사용하여 입력, 출력 데이터 값을 넣어준다

        if step % 200 == 0 : # 모든 epoch 결과를 보지 않고 20번 마다 출력해서 보기 위해 사용한 if문
            print(step, cost_val, w_val, b_val) # 훈련될 때마다 epoch 횟수와 cost, w, b를 출력

    print("COST :", cost_val) # 2000번째, 미자막 단계에서의 cost를 출력

    # 선형모델 hypothesis를 이용하여 새로운 예측값들 4 / 5, 6 / 6, 7, 8을 입력값으로 하고 어떤 결괏값이 나오는지 테스트
    print("예측(4) :", sess.run(hypothesis, feed_dict = {x_train :[4]})) 
    print("예측(5, 6) :", sess.run(hypothesis, feed_dict = {x_train : [5, 6]}))
    print("예측(6, 7, 8) :", sess.run(hypothesis, feed_dict = {x_train : [6, 7, 8]}))