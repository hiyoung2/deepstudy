import numpy as np

# 1. 데이터
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print("x.shape :", x.shape) # (10, )
print("y.shape :", y.shape) # (10, ) 

from keras.models import Sequential
from keras.layers import Dense

# 2. 모델 
model = Sequential()
model.add(Dense(1, input_dim = 1, activation = 'relu'))

# 3. 컴파일, 훈련
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs = 100, batch_size = 1)

# 4. 평가
loss, acc = model.evaluate(x, y, batch_size = 1)

print("loss :", loss)
print("acc :", acc)