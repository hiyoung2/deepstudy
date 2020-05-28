# mnist dataset 불러오고 구경하기

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist # 케라스에 예제 데이터 모음집들 중 mnist라는 걸 불러온다

(x_train, y_train), (x_test, y_test) = mnist.load_data() # mnist에서 불러온 데이터들을 x_train, y_train, x_test, y_test 자동으로 분리해준다

print('x_train : ', x_train[1])
print('y_train : ', y_train[1])

print(x_train.shape) # (60000,28,28) 60000장, height 28, width 28 / 현재 x data는 3차원
print(x_test.shape)  # (10000,28,28) 10000장, 
print(y_train.shape) # (60000, )
print(y_test.shape)  # (10000, )

print(x_train[1].shape) # x_train 중에서 index 1에 위치한 데이터 출력
plt.imshow(x_train[3], 'gray')  # index 3 데이터 gray 회색조로 보여달라
plt.show() # show 실행!하면 위에서 imshow로 부른 데이터를 볼 수 있다