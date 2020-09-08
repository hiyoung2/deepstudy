#1. 데이터 
import numpy as np
x_train = np.array([1,2,3,4,5,6,7,8,9,10])
y_train = np.array([1,2,3,4,5,6,7,8,9,10])  
x_test = np.array([11,12,13,14,15])
y_test = np.array([11,12,13,14,15])

x_pred = np.array([16,17,18]) 


#2. 모델구성         

# 과제 : R2를 음수가 아닌 0.5 이하로 줄이기
# layer는 input, output을 포함 5개 이상, node는 layer당 각각 5개 이상
# batch_size = 1
# epochs = 100 이상

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()           

model.add(Dense(4444, input_dim = 1))                                            
model.add(Dense(4444))  
model.add(Dense(4444))  
model.add(Dense(4444))  
model.add(Dense(4444))  
model.add(Dense(1))   

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)
print("loss :", loss)
print("mse :", mse)

y_pred = model.predict(x_test)

print(y_pred)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_pred):
    return np.sqrt(mean_squared_error(y_test, y_pred))

print("RMSE :", RMSE(y_test, y_pred))

# R2
from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_pred)

print("R2 :", R2)

