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
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))        

# 3. 컴파일, 훈련

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])

model.fit(x_train, y_train, epochs = 100, batch_size = 1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)

y_pred = model.predict(x_test)

print("예측값 :\n", y_pred)
print()

# RMSE

from sklearn.metrics import mean_squared_error

def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

print("LOSS :", loss)
print("MSE :", mse)
print("RMSE :", RMSE(y_test, y_pred))
print("R2 :", r2)

'''
예측값 :
 [[11.000002]
 [12.000001]
 [12.999999]
 [14.000005]
 [14.999999]]

LOSS : 2.1827872842550277e-12
MSE : 2.1827873709912016e-12
RMSE : 2.412626388678268e-06
R2 : 0.9999999999970897
'''

# R2를 음수가 아닌 0.5 이하로 줄여보기
# layer는 input, output을 포함하여 5개 이상
# node는 layer당 5개 이상
# batch_size = 1
# epochs = 100 이상