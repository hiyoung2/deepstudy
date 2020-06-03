# hite data를 pca로 차원 축소 시켜서 
# split 한 후, samsung의 data처럼 LSTM에 맞는 모델로 만들어라


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


# samsung, hite split 함수로 나눠주기

samsung = split_x(samsung, size)
print('samsung_shape : ', samsung.shape) # (504, 6, 1)

x_sam = samsung[:, 0:5] 
y_sam = samsung[:, 5] 


# scaler 먼저 무조건 써야 한다, 스케일러 다음에 pca 써야 함
x_sam = x_sam.reshape (504, 5)

scaler = MinMaxScaler()
scaler.fit(x_sam)
x_sam = scaler.transform(x_sam)

scaler = MinMaxScaler()
scaler.fit(hite)
hite = scaler.transform(hite)

x_sam = x_sam.reshape(-1, 5, 1)

# 차원 축소,  PCA
pca = PCA(n_components = 1)
pca.fit(hite)
x_hit = pca.transform(hite)


print('x_sam.shape : ', x_sam.shape) # (504, 5, 1)
print('y_sam.shape : ', y_sam.shape) # (504, 1)
print('x_hit.shape : ', x_hit.shape) # (509, 1)

# 가장 마지막에 hite를 split 함수에 넣어준다
x_hit = split_x(x_hit, size)
print('x_hit_shape : ', x_hit.shape) # (504, 6, 1)

# train_test_split
x_sam_train, x_sam_test, x_hit_train, x_hit_test, y_sam_train, y_sam_test = train_test_split(
    x_sam, x_hit, y_sam, train_size = 0.8
)

# 2. 모델 구성

input1 = Input(shape = (5, 1))
x1 = LSTM(100)(input1)
x1 = Dense(120)(x1)
x1 = Dense(150)(x1)
x1 = Dense(90)(x1)  

input2 = Input(shape = (6, 1))
x2 = LSTM(100)(input2)
x2 = Dense(150)(x2)
x2 = Dense(200)(x2)
x2 = Dense(50)(x2)


merge = Concatenate()([x1, x2]) # concatenate ([x1, x2]) / Concatenate()([x1, x2])

output = Dense(1)(merge)

model = Model(inputs = [input1, input2], outputs = output)

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit([x_sam_train, x_hit_train], y_sam_train, epochs = 10, batch_size = 32, validation_split = 0.2, verbose = 1)

# 4. 평가, 예측
loss, mse = model.evaluate([x_sam_test, x_hit_test], y_sam_test, batch_size = 32)

print("loss : ", loss)
print("mse : ", mse)

# predict 값 잡아주기(가장 최근의 것으로 가져옴)
x_sam_pred = x_sam[-1, :, :]
x_hit_pred = x_hit[-1, :, :]
x_sam_pred = x_sam_pred.reshape(-1,5,1) 
x_hit_pred = x_hit_pred.reshape(-1,6,1)

y_pred = model.predict([x_sam_pred, x_hit_pred])
print(y_pred)
