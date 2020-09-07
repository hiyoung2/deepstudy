import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

# 1. data
x = np.array(range(1, 101))
y = np.array(range(101, 201))

x_train = x[:60]
x_val = x[60:80]
x_test = x[80:]

y_train = y[:60]
y_val = y[60:80]
y_test = y[80:]

# 2. model
model = Sequential()

model.add(Dense(8, input_dim = 1))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

model.summary()

# 3. compile & fit
model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=64, batch_size=1, validation_data=(x_val, y_val))

# 4. evaluate, predict
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print("LOSS :", loss)
print("MSE :", mse)

y_pred = model.predict(x_test)
print(y_pred)

# RMSE, R2
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

rmse = RMSE(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE :", rmse)
print("R2 :", r2)


