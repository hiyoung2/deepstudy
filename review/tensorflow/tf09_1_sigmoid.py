import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2],
          [2, 3], 
          [3, 1],
          [4, 3],  
          [5, 3],
          [6, 2]]

y_data = [[0], 
          [0], 
          [0],
          [1],
          [1],
          [1]]

x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

# (1) tf.random_normal
# w = tf.Variable(tf.random_normal([2, 1]), name = 'weight')
# b = tf.Variable(tf.random_normal([1]), name = 'bias')

# (2) tf.random_uniform
w = tf.Variable(tf.random_uniform([2, 1]), name = 'weight')
b = tf.Variable(tf.random_uniform([1]), name = 'bias')

# (3) tf.zeros
# w = tf.Variable(tf.zeros([2, 1]), name = 'weight')
# b = tf.Variable(tf.zeros([1]), name = 'bias')

# sigmoid 활성화 함수 적용 방법 -> wrapping
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

cost = -tf.reduce_mean(y * tf.log(hypothesis)) + (1 - y * tf.log(1 - hypothesis))
# Cross Entropy 를 Cost(Loss) 함수(손실함수, 비용함수, 또는 목적함수)로 사용
# 1인 경우 y*tf.log(hypothesis)가 0으로 수렴하도록 학습
# 0인 경우 (1-y)*tf.log(1-hypothesis)가 0으로 수렴하도록 학습

lr  = 9e-3
optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer()) 
    
    prediction = tf.cast(hypothesis > 0.5, dtype = tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y), dtype = tf.float32))

    for step in range(1001) :
        cost_val, hy_val, _ = sess.run([cost, hypothesis, optimizer], feed_dict = {x:x_data, y:y_data})

        if step % 100 == 0 :
            print(step, "cost :", cost_val)

        acc = sess.run(accuracy, feed_dict = {x:x_data, y:y_data})

    print("acc :", acc)





        