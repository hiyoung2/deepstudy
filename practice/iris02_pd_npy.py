# iris01_save_pd.py 파일에서 저장 시켜 놓은 dataset을 불러보자

'''
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import np_utils # y data, one hot encoding 할 때 필요
from sklearn.preprocessing import MinMaxSclaer, StandardSclaer, MaxAbsScaler, RobustSclaer # 스케일러 선택은 어떻게 할까
from sklearn.model_selection import train_test_split # data 분리시 필요함
from keras.models import Sequential, Model # 시퀀셜? 함수형? 본인 자유! 시퀀셜이 훨배 편하다 ㅎㅎ 
from keras.layers import Dense, Dropout, LSTM, Conv2D, Maxpooling2D, Flatten # 어떤 모델을 짜느냐에 따라 사용하는 것이 다름, 골라 쓰자
                                                                             # 다 소환해서 쓸 것 써도 된다(대신 전체 파일의 용량 크기가 좀 커진다)
from keras.callbacks import EarlyStopping, ModelCheckpoint, Tensorboard # 배운 것 다 써먹어야한다

# 1. 데이터 준비

dataset = np.load('./data/iris.npy')
# 필요한 것들은 다 불러왔으니 이제 데이터를 불러와서 본격적인 모델 구성 시작
# dataset은 변수명, 통상적으로 알아 먹을 수 있는 이름을 사용하자
# np 형태로 load 한다, data 폴더 내의 iris.npy 파일을!

print(dataset) # 데이터 살펴보기

# 지금 iris 데이터를 살펴 보면
# index 0부터 3까지는 x data에 해당(독립변수? 원인, 입력값)하는 것 같고
# idnex 4는 y data에 해당(종속변수? 결과, 효과)하는 것 같다
# index 4 데이터들을 살펴 보면 0과 1, 2로 이루어져있다
# 이것은 결괏값이 0 아니면 1 아니면 2란 말 즉, 세 가지 결괏값이 나올 수 있는 분류 모델 중 다중분류를 말한다
# 따라서 모델을 구성 할 때 categorical_crossentropy를 써야 하고, 마지막 output layer에서는 'softmax' 함수를 써 줘야 함을 기억해야 한다
# 그런데 지금 index 4, 하나의 column에 결괏값들이 위치해 있으므로 
# 이것을 다중분류 모델에서 쓰기 위해서는 np_utils의 to_categorical로 one-hot encoding 과정을 거쳐
# y data를 정리해줘야 한다

# 일단 index 0, 1, 2, 3에 해당하는 4개의 column은 x data로 슬라이싱, 나머지는 y data로 슬라이싱 해 줘야 한다

x = dataset[:, :4] # dataset의 모든 행에서 index[3]의 컬럼까지 x로 사용 
y = dataset[:, 4:] # dataset의 모든 행에서 index[4]의 컬럼을 y로 사용

print('x.shape : ', x.shape)
print('y.shape : ', y.shape)
# x와 y의 데이터가 어떤 와꾸를 가졌는지 체크

# 그 다음, 모델에 집어넣기 전에 x와 y data 전처리가 필요하다
# y는 원앤핫인코딩(분류모델), x는 MinMaxSclaer(분류모델에선 이걸 통상적으로 쓰는 것 같다?)

y = np_utils.to_categorical(y)
print('y.shape : ', y.shape)

sacler = MinMaxScaler()
scaler.fit(x) # scaler를 실행한다
x = scaler.transform(x)  # sclaer를 통해 transform된 형태를 x에 대입
                         # 정규화를 거쳐 x data들의 형태가 변함
                         # minmax를 썼기 때문에 0과 1 사이의 숫자들로 x data가 변신


# x와 y 각각 데이터들의 전처리를 마쳤음
# 그러면 훈련, 평가에 들어가기 위한 데이터 분리를 해야 한다

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.8, random_state = 77, shuffle = True
)

# 전체 데이터에서 80%를 train data로 사용한다, shuffle을 하면 아무래도 더 좋은 결과를 볼 수 있을 것

# 데이터 불러오기, 데이터 슬라이싱, 데이터 전처리, 데이터 분리를 모두 끝마쳤으니 본격적인 모델 구성에 들어간다

# 2. 모델 구성
model = Sequential() # 시퀀셜 형 사용! (함수형이랑 성능적으론 차이가 없어 보인다, 본인이 편한 것 쓰면 되나??)

model.add(Dense(50, input_shape = (4, ))) # 가장 중요한 와꾸 맞추기! Dense 모델은 x data의 column 수를 주의해주면 된다
model.add(Dense(100))
model.add(Dense(3, activation = 'softmax')) # 여기서도 와꾸 맞추기! y data의 결괏값 종류가 3가지이므로 ouput을 3으로 맞춰준다
                                            # 그리고 다중분류 모델에서는 softmax 함수를 마지막에 써 준다(이유는? 더 공부해야함)
                                            # 일단 주입식처럼 이중분류는 sigmoid(0아니면1), 다중분류는 softmax 라고 알고 있자

model.summary() # 전체적인 모델 요약본 확인

# 3. 컴파일, 훈련

# 배운 기능들을 다 써보자(callbacks에 있는 것들, 특히)

es = EarlyStopping(monitor = 'loss', patience = 10, mode='auto') # mode는 auto 해 주면 된다 그냥!

ckpath = './model/어쩌구/저쩌구위치/iris_check-{epoch:02d}-{val_loss:.4f}.hdf5'
# 체크포인트가 저장되는 위치를 설정, 그리고 파일명 설정, hdf5의 파일형식을 가진다
# epoch는 2자리 정수 모양으로 표시하고 val_loss는 소숫점 넷째자리까지 보여준다고 설정

checkpoint = ModelCheckpoint(filepath = ckpath, monitor = 'val_loss', save_best_only = True, save_weights_only = False, verbose = 1)
# ModelCheckpoint를 사용하기 위해 checkpoint라는 변수를 만들었다
# 저장되는 파일의 경로를 써 주고(ckpath라는 변수에 위치, 파일명이 담겨 있다)
# val_loss를 관찰할 것이고, 가장 좋은 것들만 나타내달라고 지정(save_weights_only, 가중치 저장한 것은 안 보여줘도 된다고 False로 지정)
# verbose는 일단 넣음

tb_hist = Tensorboard(log_dir = 'graph', histogram_freq = 0, write_graph = True, write_images = True)
# tb_hist는 텐서보드를 실행시키기 위한 변수명
# graph라는 폴더(미리 만들어둠)에 fit 과정을 그래프로 볼 수 있다


model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrcis = ['acc'])
# 모델을 컴파일 한다, 다중분류이므로 categorical_crossentropy를 써 주고 최적화 함수로는 adam(평타 85), 훈련 방식을 acc로 보고 싶다고 설정
# 분류모델이므로 acc로 판단

hist = model.fit(x_train, y_train, epochs = 100, batch_size = 32, callbacks = [es, checkpoint, tb_hist], validation = 0.2, verbose = 1)
# hist에 fit 과정을 대입
# fit, 훈련 실행, callbacks에 사용한 것들을 다 적어줘야 적용, 실행된다

print(hist.history.keys()) # model fit에서 반환되는 것이 hist에 저장되었는데 
                           # 이 프린트문을 실행하면 무엇의 history를 볼 수 있는지 알 수 있다

# tensorboard는 웹사이트에서 실행 과정의 그래프를 볼 수 있고
# matplotlib.pyplot , plt를 통해서는 여기서 바로 그래프를 볼 수 있고, 직접 그래프를 꾸밀 수 있다

# 시각화

plt.figure(figsize = (10, 6))

plt.subplot(2, 1, 1) # 2가지 그림을 볼 건데 2행 1열의 형태로 나타낼 것이고 그 중 1번째 그림을 지칭
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss') # loss 그래프를 . 표시, 빨간 색으로 꾸미고 loss라는 라벨을 붙인다
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss') # 한 사진에서 loss뿐만 아니라 val_loss도 함께 나타냄
plt.grid() # 그래프에 격자? 무늬를 넣음
plt.title('loss') # 그래프의 제목
plt.ylabel('loss') # y축은 loss
plt.xlabel('epoch') # x축은 epoch
plt.legend(loc = 'upper right') # 우측 상단에 loss, val_loss 그림 주석? 설명? 위치

plt.subplot(2, 1, 2) 2행 1열 형태에서 2번째 그림
plt.plot(hist.hitory['acc'], marker = '*', c = 'green', label = 'acc')
plt.plot(hist.history['val_acc'], marker = "*", c = 'purple', label = 'val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')
plt.show() # 그래프를 보여달라!(이건 한 번만 써야 한다, 저번에 위 아래 둘 다 썼더니 그림이 두 개로 따로 나타났었음)

'''