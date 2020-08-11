from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7,8,9,10])  
y_train = np.array([1,2,3,4,5,6,7,8,9,10])
x_test = np.array([101,102,103,104,105,106,107,108,109,110])
y_test = np.array([101,102,103,104,105,106,107,108,109,110])

print("x_train.shape :", x_train.shape) # (10,)
print("y_train.shape :", y_train.shape) # (10,)
print("x_test.shape :", x_test.shape) # (10,)
print("y_test.shape :", y_test.shape) # (10,)

# 2. 모델
model = Sequential()
model.add(Dense(8, input_dim = 1, activation = 'relu'))
model.add(Dense(16))
model.add(Dense(32))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1, activation='relu'))

model.summary()

'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 8)                 16
_________________________________________________________________
dense_2 (Dense)              (None, 16)                144
_________________________________________________________________
dense_3 (Dense)              (None, 32)                544
_________________________________________________________________
dense_4 (Dense)              (None, 64)                2112
_________________________________________________________________
dense_5 (Dense)              (None, 128)               8320
_________________________________________________________________
dense_6 (Dense)              (None, 64)                8256
_________________________________________________________________
dense_7 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_8 (Dense)              (None, 16)                528
_________________________________________________________________
dense_9 (Dense)              (None, 8)                 136
_________________________________________________________________
dense_10 (Dense)             (None, 1)                 9
=================================================================
Total params: 22,145
Trainable params: 22,145
Non-trainable params: 0
_________________________________________________________________
'''

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

model.fit(x_train, y_train, epochs=100, batch_size=1)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=1)
print("loss: ", loss)
print("acc :", acc)

output = model.predict(x_test)
print("결과물 : \n", output)

'''
loss:  2.735760062932968e-10
acc : 1.0
결과물 :
 [[101.00001 ]
 [102.000015]
 [103.00003 ]
 [104.000015]
 [105.00002 ]
 [106.000015]
 [107.00001 ]
 [108.00001 ]
 [109.000015]
 [110.00001 ]]
'''