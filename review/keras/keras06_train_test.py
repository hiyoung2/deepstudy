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
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)

print("loss :", loss)
print("mse :", mse)

y_pred = model.predict(x_pred)

print("예측값 :\n", y_pred)

'''
loss : 1.0913936421275138e-12
mse : 1.0913936854956008e-12
예측값 :
 [[16.000002]
 [16.999998]
 [18.000002]]
'''