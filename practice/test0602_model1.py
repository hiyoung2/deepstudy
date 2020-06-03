# ensemble로 모델을 짜야 한다

import numpy as np
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.layers import concatenate, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# CNN이 LSTM 보다 시계열에 더 좋은 경우도 있다
# CNN은 LSTM 보다 빠른 속도로 처리 가능하다

def split_x(seq, size):                          
    aaa = []                                        
    for i in range(len(seq) - size + 1):            
        subset = seq[i : (i+size)]                  
        aaa.append([j for j in subset])       
                                                   
    # print(type(aaa))                                
    return np.array(aaa)   

size = 6

# 1. 데이터
# npy 불러오기

samsung = np.load('./data/samsung.npy',  allow_pickle = True)
hite = np.load('./data/hite.npy', allow_pickle = True)

print('samsung_shape : ', samsung.shape) # (509, 1)
print('hite_shape : ', hite.shape)       # (509, 5)

samsung = samsung.reshape(samsung.shape[0], )
print('samsung.shape : ', samsung.shape) # (509, ) 으로 reshape 되었다

# samsung = split_x(samsung, size)
# print('samsung_shape(noreshape) : ', samsung.shape) # (504, 6, 1) -> 바로LSTM을 쓸 거면 적합하나, Dense 모델을 쓰려면 reshape을 해 줘야 한다 
                                                      # 컬럼 하나 짜리는 애초에 벡터 형태로 바꿔주는 것이 편하다
                                                      # 벡터 형태로 바꿔주고 모델 구성에 따라 reshape 해주자

samsung = split_x(samsung, size)
print('samsung_shape(reshape) : ', samsung.shape) # (504, 6) # 5일치를 x, 다음 하루치를 y 따라서 6이 됨

# 삼성만 x, y 분리하면 됨
# 삼성 주가 예측하는 거니까 하이트는 y로 해 줄 필요가 없다(x로 쓰면 됨)

x_sam = samsung[:, 0:5]
y_sam = samsung[:, 5] 

print(x_sam)
print(y_sam)


print('x_sam.shape : ', x_sam.shape) # x_sam.shape :  (504, 5)
print('y_sam.shape : ', y_sam.shape) # y_sam.shape :  (504, )

# 주식은 최근 것을 반영해주는 게 좋다
x_hit = hite[5:510, :]
print('x_hit.shape : ', x_hit.shape) #  (504, 5)


# train_test_split
x_sam_train, x_sam_test, x_hit_train, x_hit_test, y_sam_train, y_sam_test = train_test_split(
    x_sam, x_hit, y_sam, train_size = 0.8
)

# 2. 모델 구성

input1 = Input(shape = (5, ))
x1 = Dense(10)(input1)
x1 = Dense(10)(x1)

input2 = Input(shape = (5, ))
x2 = Dense(5)(input2)
x2 = Dense(5)(x2)

merge = concatenate([x1, x2])

output = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = output)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x_sam_train, x_hit_train], y_sam_train, epochs = 100, batch_size = 32)

# 4. 평가, 예측
loss, mse = model.evaluate([x_sam_test, x_hit_test], y_sam_test, batch_size = 32)

print("loss : ", loss)
print("mse : ", mse)


# 현재 hite는 (509, 5), x_sam은 (504, 5)
# 열은 맞는데 행은 맞지 않다
# fit 하니까 에러메세지 발생
# ValueError: All input arrays (x) should have the same number of samples. Got array shapes: [(504, 5), (509, 5)]
# 앙상블 모델은 행을 맞춰줘야 한다
# hite의 행을 잘라내는 방법밖에? x_hit로 가시오!

