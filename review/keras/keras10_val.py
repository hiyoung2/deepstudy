#1. data
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])  
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])


x_val = np.array([101, 102, 103, 104, 105]) 
y_val = np.array([101, 102, 103, 104, 105])

print(x_train.shape) # (10, )
print(x_test.shape) # (5, )

x_train = x_train.reshape(10, 1)
x_test = x_test.reshape(5, 1)

# 2. model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(8, input_dim = 1))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))


model.summary()

print("==========================================================")

# 3. compile & fit
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=64, batch_size=1, validation_data=(x_val, y_val))


# 4. evaluate & predict
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
y_pred = model.predict(x_test)

print("LOSS :", loss)
print("MSE :", mse)
print("예측값 :", y_pred)
print("==========================================================")

# 5. RMSE & R2
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE :", RMSE(y_test, y_pred))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R2 :", r2)
