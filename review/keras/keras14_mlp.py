# mlp : multi layer perceptron
# 가장 기본적인 형태의 인공신경망(Artificial Neural Network) 구조이며
# 하나의 입력층(input layer)
# 하나 이상의 은닉층(hidden layer)
# 그리고 하나의 출력층(output layer)로 구성된다

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터

x = np.array([range(1,101), range(311,411), range(100)])
y = np.array([range(101,201), range(711,811), range(100)]) 

"""
x range(1,101)
y range(101,201)
w=1, bias=100

x range(311,411)
y range(711,811)
w=1, bias=400

x range(100) 은 0~99 
y range(100) 은 0~99 
w=1, bias=0
"""

print(x)
print("x.shape :", x.shape) # x.shape : (3, 100)
print()

# rows <-> columns
x = np.transpose(x)
y = np.transpose(y)

print(x)
print()
print(y)

print("x_trans.shape :", x.shape) # (100, 3)
print("y_trans.shape :", y.shape) # (100, 3)

# split data(train/test)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 77, train_size = 0.8
)

print("x_train.shape :", x_train.shape) # (80, 3)
print("x_test.shape :", x_test.shape) # (20, 3)

# 2. model
model = Sequential()
model.add(Dense(10, input_dim = 3))                                            
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(3))


# 3. train (compile & fit)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.3)

# 4. evluate & predict
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
y_pred = model.predict(x_test)

print("loss :", loss)
print("mse :", mse)
print()
print("prediction :", y_pred)
print()

# RMSE, R2

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

R2 = r2_score(y_test, y_pred)

print("RMSE :", RMSE(y_test, y_pred))
print("R2 :", R2)
