import numpy as np
import tensorflow as tf
tf.set_random_seed(777)

dataset = np.loadtxt('./data/data-01-test-score.csv', delimiter = ',', dtype = np.float32)

x_data = dataset[:, :-1]
y_data = dataset[:, -1:]

print("x_data.shape :", x_data.shape)
print("y_data.shape :", y_data.shape)

x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(x, w) + b

# 손실함수 : mse
cost = tf.reduce_mean(tf.square(hypothesis - y))

lr = 5e-5
optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(cost)

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(2001) :
        cost_val, hy_val, _ = sess.run([cost, hypothesis, optimizer], feed_dict = {x:x_data, y:y_data})

        if step % 100 == 0 :
            print(step, "cost :", cost_val)
            print("예측값 :", hy_val)