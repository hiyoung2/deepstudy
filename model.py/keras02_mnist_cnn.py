import numpy as np
import matplotlib.pyplot as plt

# 나름의 순서를 정해서 외우자
from keras.datasets import mnist # 데이터 준비 위해 필요
from keras.utils import np_utils # 데이터 전처리를 위해 필요
from keras.models import Sequential # 모델 구성시, 첫 번째 선택 사항
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # 모델 구성시, 두 번째 선택 사항


# 1. 데이터 준비
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # (60000, 28, 28)
print(x_test.shape) # (10000, 28, 28)
print(y_train.shape) # (60000,)
print(y_test.shape) # (10000,)

# 1.1 데이터 전처리, one hot incoding (y data)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train.shape) # (60000, 10)

# 1.2 데이터 전처리, Scaler (x data)
# 4차원으로 reshape(CNN), type 실수형으로 바꿔서 255로 나누는 것은 MinMax Scaler 와 거의 같은 효과를 내기 위한 전처리 방법 중 하나
# 타입을 실수형으로 바꿔서 255로 나누면 모든 데이터들이 0 ~ 1 사이 범위에 있게 되기 때문에 MinMax와 거의 비슷한 방법이라 한다
# 원래 타입은 int형, 0 ~ 255(0 : 흰색, 255 : 완전 찐한 검정색)
x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.0 
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 225.0
print(x_train.shape) # (60000, 28, 28, 1) 4차원으로 변신

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(28, (2, 2), input_shape = (28, 28, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(28, (3, 3), activation = 'relu', padding = 'same'))
model.add(Dropout(0.2))
model.add(Conv2D(28, (4, 4), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

# 3. 컴파일, 훈련
from keras.callbacks import EarlyStopping # 훈련 시 필요한 사항
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'max')

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, epochs = 50, batch_size = 200, validation_split = 0.2, callbacks = [early_stopping], verbose = 1)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 200)

print('loss : ', loss)
print('acc : ', acc)

y_pred = model.predict(x_test)
print(y_pred)
print(np.argmax(y_pred, axis = 1))