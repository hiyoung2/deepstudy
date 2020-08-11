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
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.fit(x, y, epochs = 100, batch_size = 1)

# 4. 평가, 예측
loss, acc = model.evaluate(x, y, batch_size = 1)

print("loss :", loss)
print("acc :", acc)

y_pred = model.predict(x_pred)
print("x_pred의 예측값 : \n", y_pred)

'''
loss : 3.5385028240852987e-13
acc : 1.0
x_pred의 예측값 :
 [[11.      ]
 [11.999999]
 [13.000001]]
'''