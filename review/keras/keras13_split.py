# 0. import
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score


# 1. data

x = np.array(range(1, 101))
y = np.array(range(101, 201))

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state = 77, shuffle = True, train_size = 0.6
)

# 2. model
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

# 3. train(compile & fit)
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 200, batch_size = 1, validation_split = 0.3)

# 4. evaluate & predict
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss :", loss)
print("mse :", mse)
print()

y_pred = model.predict(x_test)
print("prediction :", y_pred)
print()

# RMSE & R2
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

R2 = r2_score(y_test, y_pred)

print("RMSE :", RMSE(y_test, y_pred))
print("R2 :", R2)