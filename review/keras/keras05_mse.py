import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])    
x_pred = np.array([11,12,13]) 

# 2. 모델 

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

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x, y, batch_size=1)

print("loss :", loss)
print("mse :", mse)

y_pred = model.predict(x_pred)
print("예측값 :\n", y_pred)


'''
loss : 2.1174173525650986e-13
mse : 2.11741729835499e-13
예측값 :
 [[10.999999]
 [11.999999]
 [13.000001]]
'''