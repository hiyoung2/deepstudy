import numpy as np

from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터

x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])  
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

x_pred = np.array([16,17,18])

# 2. 모델

model = Sequential()           

model.add(Dense(8, input_dim = 1))  
model.add(Dense(16))                
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))      

# 3. 평가, 에측
model.compile(loss = 'mse', optimizer = 'adam', metrics= ['mse'])

model.fit(x_train, y_train, epochs = 100, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)



y_pred = model.predict(x_test)
print("예측값 :\n", y_pred)

# RMSE 
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred) :

    return np.sqrt(mean_squared_error(y_test, y_pred))


print("loss :", loss)
print("mse :", mse)
print("RMSE :", RMSE(y_test, y_pred))

'''
 [[11.000002]
 [12.000002]
 [13.000002]
 [14.000002]
 [15.000001]]
loss : 5.638867150992155e-12
mse : 5.6388669775198075e-12
RMSE : 1.7584885515771651e-06
'''