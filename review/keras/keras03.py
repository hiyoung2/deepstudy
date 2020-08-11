import numpy as np

# 1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])    

# 2. 모델
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_dim = 1))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.fit(x, y, epochs=30, batch_size=1)

# 4. 평가, 에측
loss, acc = model.evaluate(x, y, batch_size=1)

print("loss :", loss)
print("acc :", acc)

'''
loss : 0.00021722162166497582
acc : 1.0x
'''