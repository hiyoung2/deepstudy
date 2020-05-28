import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 1. 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  # (60000, 28, 28)
print(x_test.shape)   # (60000, 28, 28)
print(y_train.shape)  # (60000,)
print(y_test.shape)   # (10000,)

# 1.1 데이터 전처리, one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10)

# 1.2 데이터 전처리, scaling
x_train = x_train.reshape(60000, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(10000, 28*28).astype('float32') / 255.0
print(x_train.shape) # (60000, 784)

# 2. 모델 구성
model = Sequential()

# model.add(Dense(100, input_shape = (28*28, )))
# model.add(Dropout(0.2), activation = 'relu')
# model.add(Dense(50), activation = 'relu')
# model.add(Dropout(0.2), activation = 'relu')
# model.add(Dense(10))

# model.summary()
'''
# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'max')

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size = 100, validation_split = 0.3, callbacks = [early_stopping], verbose = 1)


# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 100)
print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test)
print(y_pred.shape)
'''