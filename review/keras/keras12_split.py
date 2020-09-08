# 1. data
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(101, 201))

# train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 77, shuffle = True, train_size = 0.8
)

x_val, x_test, y_val, y_test = train_test_split(
    x_test, y_test, random_state = 77, shuffle = True, test_size = 0.4
)

print("x_train.shape :", x_train.shape)
print("x_val.shape :", x_val.shape)
print("x_test.shape :", x_test.shape)
# x_train.shape : (80,)
# x_val.shape : (12,)
# x_test.shape : (8,)

# 2. model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()

model.add(Dense(8, input_dim=1))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

model.summary()

# 3. compile & fit
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=128, batch_size=10, validation_data=(x_val, y_val))


# 4. evaluate, predict
loss, mse = model.evaluate(x_test, y_test, batch_size=10)

y_pred = model.predict(x_test)

print("LOSS :", loss)
print("MSE :", mse)
print("예측값 :", y_pred)

# RMSE, R2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse = RMSE(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE :", rmse)
print("R2 :", r2)