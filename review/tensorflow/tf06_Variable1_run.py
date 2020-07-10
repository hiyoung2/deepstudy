import tensorflow as tf
tf.set_random_seed(777)

# 지금까지 w, b를 설정했던 방법 , Variable이라는 변할 수 있는 값으로 설정했었음
w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# w 값 변화 시켜주기(Variable이므로 설정헀다고 고정되지 않음)

w = tf.Variable([0.3], tf.float32)

# w 에 어떤 값이 들어있는지 확인(feat.서인국)
# 아래의 3가지 방법이 있다
# 문법적인 부분들이 다른 거고, 기능은 같은 것들이다

# 1. 수업시간 가장 처음 배운 방법
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 선언!(초기화), Variable을 사용한다면 꼭 해야 하는 절차
aaa = sess.run(w)
print("weight 확인 :", aaa) # weight 확인 : [0.3]
sess.close()
print('----------------------------------------------------')


# 2. eval을 사용하는 방법
sess = tf.Session()
sess.run(tf.global_variables_initializer())
bbb = w.eval(session = sess) # session = sess를 명시해줘야 한다! (Session으로 해봤더니 안 됨)
print("weight 확인 :", bbb) # weight 확인 : [0.3]
sess.close()
print('----------------------------------------------------')


# 3. IntereactiveSession과 eval을 사용한 방법
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
ccc = sess.run(w)
print("weight확인 :", ccc) # weight 확인 : [0.3]
sess.close()
print('----------------------------------------------------')

