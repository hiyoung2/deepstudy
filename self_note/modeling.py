# # 모델 구성 순서

# # 0. import
import numpy as np
# numpy를 불러오고, 앞으로 np라고 부른다
# numpy에 관해서는 공부가 더 필요하다

# import matplotlib.pyplot as plt
# matplotlib.pypplot을 불러오고 plt라 부른다
# matplotlib은 파이썬에서 시각화를 처리하는데 필요한 대표적인 라이브러리이다
# 이미지를 처리한다고 할 때, 중간중간 결과를 확인하기 위해서 matplotlib.pyplot으로 해당 이미지를 시각화 할 수 있다
# 수업시간에는 loss와 mse(또는 acc)의 추이를 관찰하기 위해서 이용했다
# loss와 mse, acc등이 그래프를 통해서 나타나는데 이 때, 색깔, 레이블에 대한 설명, 모양 등을 설정할 수 있다


from keras.datasets import mnist
# 케라스에서 제공하는 예제 데이터셋을 불러오는 방법이다
# mnist라는 예제를 부른다

from sklearn.datasets import load_iris
# 싸이킷런에서 제공하는 예제 데이터셋을 불러오는 방법이다
# 케라스랑은 다르게 예제명 앞에 'load_'를 써 줘야 한다

# from keras.utils import np_utils 
# 케라스에서 제공하는 one-hot encoding을 수행하는 도구를 불러오기 위한 것
# 이렇게 불러오면 데이터 전처리 과정에서
# y = np_utils.to_categorical(y) 라고 써 줘야 하는데 
# 아래와 같이
# from keras.utils import to_categorical
# y = to_categorical(y)
# 이런 식으로 써 줘도 된다


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
# 싸이킷런에서 데이터 전처리를 위해 제공하는 도구들
# MinMax와 Standard가 가장 대표적인 두 가지

# 1) StandardScaler 
# - 각 feature의 평균을 0, 분산을 1로 변경한다. 모든 특성들이 같은 스케일을 갖게 된다
# - 평균을 제거하고 데이터를 단위 분산으로 조정한다. 그러나 이상치가 있다면 평균과 표준편차에 영향을 미쳐 변환된 데이터의 확산은 매우 달라지게 된다.
# - 따라서 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다. 
# -> 이상치를 제거하는 방법????

# ex) 데이터 1 2 3 4 10 이 있다고 치자
# 데이터의 평균 : 4
# 편차 : 평균 - 각 데이터 값 , 3, 2, 1, 0, -6
# 편차의 제곱 : 9, 4, 1, 0, 36
# 편차의 제곱 평균 : 9 + 4 + 1 + 0 + 36 / 5 = 10(분산)
# 표준편차 : 루트분산, 루트10 = 대략 3.3이라 치자
# 데이터 값 - 평균 / 표준편차 
# 1 - 4 / 3.3 = -1 정도
# 10 - 4 / 3.3 = 2 정도
# 데이터들이 -1에서 2사이에 분포


# 2) MinMaxScaler
# - 모든 feature가 0과 1 사이에 위치하게 만든다
# - x - xmin / xmax - xmin 
# - 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다
# = 역시 이상치, outlier의 존재에 매우 민감하다

# 3) MaxAbsScaler
# - 절댓값이 0 ~ 1 사이에 위치하도록 한다, 즉 -1 ~ 1 사이로 재조정
# - 양수 데이터로만 구성된 데이터셋에서는 MinMaxScaler와 유사하게 동작
# - 큰 이상치에 민감

# 4) RobustScaler
# - 이상치, outlier의 영향을 최소화한 기법
# - 중앙값(median)과 IQR(interquartile range)을 사용한다
# - StandardScaler와 비교하면 표준화 후 동일한 값을 더 넓게 분포시킨다

# 결론적으로는 스케일러 처리 전에 이상치 제거는 선행 되어야 한다
# 그리고 데이터의 분포 특징에 따라 적절한 스케일러를 적용해주어야 한다
# 어떻게???


from keras.models import Seuqential, Model
# 데이터도 불러오고 전처리까지 마쳤다면 본격적인 모델 구성에 들어간다
# add 하면서 순차적으로 모델을 구성하는 Sequential 형식이 있고
# input layer, output layer 안에 꽁지를 무는 모양으로 구성하는 함수형 모델, Model 형식도 있다
# Sequential형은 model = Seuqeuntial()를 모델 구성에 앞서 적어줘야 한다
# Model형은 모델 구성을 한 뒤에 model = Model(inputs = input1, outputs = output1)를 모델 구성 마무리에 적어줘야 한다
# 무엇을 사용하든 내 마음!
# 개인적ㄹ으로, Sequential 형이 간단하게 작성 가능해서 편하다


from keras.layers import Dense, Dropout 
# DNN 모델을 만들 때 필요한 레이어들을 케라스로부터 불러들인다
# Dense와 Dropout
# Dense로만 구성하는 모델은 가장 간단!
# Dense는 2차원으로 input_shape에는 (10, ) 이런 식으로 들어간다(x data의 column : 10)
# Dropout은 학습 과정에서 신경망의 일부를 사용하지 않는 방법으로
# 어느 모델에서건 사용 가능하다
# 예를 들어, model.add(Dropout(0.5)) 는
# 학습 과정마다 랜덤으로 50%의 뉴런을 사용하지 않는다는 뜻이다
# Dropout은 신경망 학습 시에만 사용하고 예측 시에는 사용하지 않는 것이 일반적
# 학습 시에 인공 신경망이 특정 뉴런 또는 특정 조합에 너무 의존적이게 되는 것을 방지해주고
# 매번 랜덤 선택으로 뉴런들을 사용하지 않으므로 
# 서로 다른 신경망들을 앙상블하여 사용하는 것과 같은 효과를 내어 과적합을 방지한다


from keras.layers import LSTM, Dense, Dropout
# LSTM 모델을 만들 때 필요한 레이어들을 케라스로부터 불러들인다
# DNN 모델을 만들 때 사용한 레이어들을 포함, LSTM이란 레이어를 같이 불러온다
# LSTM은 Long Short Term Memory의 약자
# RNN Model에서 주로 사용, simpleRNN 모델보다 4배의, GRU보다는 3배의 연산을 더 한다는 특징이 있다
# LSTM parameter : 4 * (input_dim + bias + output) * output
# GRU parameter : 3 * (input_dim + bias + output) * output
# SimpleRNN paramete : (input_dim + bias + output) * output
# 4라는 숫자는 연산 과정에 4개의 문을 사용한다고 일단은 그렇게 이해하고 있으면 된다
# forget gate, cell state, input gate, output gate
# LSTM 모델은 '자기 회귀'의 성질을 가지고 있다고 생각하자 (역전파.....?에 대한 공부 필요)

# LSTM layer를 첫 번째 레이어로 사용, 두 번째에도 LSTM 레이어를 사용하려면
# 첫 번째 레이어에서 'return_sequences = True' 를 함께 써 주어야 한다
# ex) model.add(LSTM(30, return_sequences = True, activation = 'relu)) 이런 식으로!
# LSTM은 3차원으로 input_shape에는 2차원 형식으로 써 준다 ex) (10, 1)
# 다음 레이어를 LSTM으로 쓰려면 차원을 맞춰줘야 하는데 차원을 유지시켜주는 게 'return_sequences'라고 알고 있으면 된다
# 역시 Dropout 사용 가능!
# LSTM 레이어를 5개 붙이는 실습을 했었는데 시간이 굉장히 오래 걸렸었음 (CPU에서)
# LSTM은 3차원, (batch, timesteps, feature)로 구성
#                 행       열       몇 개씩 자를지? -> 무조건 기억!

from keras.layers import Conv2D, Maxpooling2D, Flatten, Dense, Dropout
# CNN 모델을 구성하기 위해 필요한 레이어들
# 일단 실습시간에 사용한 것은 Conv2D를 기본으로 했다
# model.add(Conv2D(11, (2, 2), input_shape = (2, 2, 1))) 이런 식으로 사용한다
# CNN 모델은 4차원(혹은 그 이상), input_shape는 height, width, feature(?, 1이면 흑백 3이면 컬러)로 구성
# (2, 2)는 필터 또는 커널(이미지의 특징을 찾아내기 위한 공용 파라미터, 일반적으로 (2, 2), (3, 3)과 같이 정사각 행렬로 정의됨)
# Maxpooling2D로 특정 영역의 최댓값인 것들을 뽑아준다고 생각하면 된다
# model.add(Maxpooling2D(pool_size = 2)) 이런 식으로 쓰는데 2x2fh 나눠서 제일 특성 높은 놈들을 골라준다
# Flatten은 Dense layer 들어가기 전 겹쳐 있는 데이터들을 하나로 쫙 펼쳐주는 역할
# CNN 모델에서는 padding이라는 게 있는데 Convlution layer의 출력 데이터가 줄어드는 것을 방지하는 방법이다
# 입력 데이터의 외곽에 지정된 픽셀만큼 특정 값을 채워 넣는 것을 의미한다
# 예시 : model.add(Conv2D(10, (2, 2), padding = 'same'))

# 공통 : 함수형, Model을 쓸 거면 Input layer를 각각 추가해줘야 한다

from keras.callbacks import EarlyStopping, ModelCheckpoint, Tensorboard
# fit 과정에서 사용하는 것들
# EarlyStopping : 조기학습종료, monitor, patience, mode
# ModelCheckpoint : 먼저 파일경로, 파일명 지정한 후, filepath, monitor, save_best_only, mode 설정
# tensorboard : 웹상에서 loss , mse(acc)의 그래프, 모델 전개도(?)를 볼 수 있다
# tb_hist(변수명) = Tensorboard(log_dir = 'graph), histogram_freq = 0, write_graph = True, write_images True
# hist = model.fit(~~), 그래프 폴더 생성 - cmd창에서 d: 입력 = cd study 입력 = cd graph 입력 = tensorboard ==logdir=. 입력
# 주소 하나 생성되는데 인터넷 주소창에 복붙해서 들어가면 그림들을 볼  수 있다


# # 1. load data
# 케라스 : (x_train, y_train), (x_test, y_test) = mnist.load_data() : train, test data 자동 분리
# 싸이킷런 : dataset은 iris dataset을 저장 시켜 놓은 상자, 즉 변수라고 생각하면 될 듯?
#           iris_data, iris, iris_dataset 등등 마음대로 해도 되는 대신 x, y 넣을 때 통일 하는 것에만 주의하면 될 듯
#           x는 data, y는 target으로 잡아주는 것이 싸이킷런에서 데이터 불러올 때의 특징
# dataset = load_iris()
# x = dataset['data']
# y = dataset['target]


# 2. check the data shape
# 불러온 데이터들이 어떤 구조를 갖고 있는지 확인해 봐야 한다

# 3. data preprocessing
# 스케일링(scaler), PCA 등등을 통해 데이터 전처리 과정이 필요하다
# 이상치, 결측치 등에 관해서 처리를 해야 한다
# 스케일러 중에선 어떤 걸 써야하는지, PCA를 쓰는 것이 좋은 건지 등등에 대해선 아직 잘 모름
# 분류 모델에 들어갈 때는 y data는 one-hot-encoding 과정이 필요하다

# 4. train data, test data split
# 전체 data 중 train data, test data로 분리가 필요하다
# keras에서 불러 온 dataset은 따로 split 할 필요가 없다
# split 할 때, shuffle을 사용하자

# 5. modeling
# 모델 구성하기 전, 꼭 해당 모델에 맞게 shape를 맞춰주자
# 데이터 크기에 따라 레이어 구성, 노드 갯수 설정 등을 어떻게 할 지 생각해야 한다
# Dropout을 적극 활용해보자


# 6. check the model summary
# 총 파라미터 갯수를 확인할 수 있다

# 7. compile
# loss : mse(회귀), bianry_crossentropy(분류, 이진) categorical_crossentropy(분류, 다중)
# optimizer : 현재 adam만 사용중(평타 85?)
# metrics : mse(회귀), acc(분류)

# 8. fit
# machine 훈련 실행
# x_train, y_train, epochs 횟수, batch_size 설정, validation_split 설정, verbose 설정, callbacks 로 필요한 것 사용하기

# 9. evaluate
# loss와 mse(또는 acc)로 test data에 대한 평가를 내린다
# lsos와 mse(또는 acc)를 print 해서 살펴 본다
# loss, mse = model.evaluate(x_test, y_test, batch_size = ?)

# 10. predict
# 예측 데이터로 주어진 x_pred로  y_pred 를 예측
# y_pred = model.predict(x_pred)
# x_pred 값이 주어지지 않았을 때 x_test로 예측 했었음
# softmax 함수 사용시, argmax를 사용해 가장 큰 값을 추출해내야 한다
# ex) print(np.argmax(y_pred, axis = 1))
# axis 는 축으로 axis = 0은 열, axis = 1은 행이다
# 행을 기준으로 가장 큰 값들을 뽑아내 달라는 주문

# 11. visualization
# 모델의 결과(?)를 시각화해서 볼 수 있다
# matplotlib.pyplot을 통해!
# plt.figure(figsize = (10, 6)) : 출력되는 그림의 사이즈를 지정(inch)
# plt.subplot(2, 1, 1) 그림이 두 가지인 경우 2행 1열의 형태로 첫 번째 그림이라는 말
# plt.plot(hist.history['loss'], marker = '.', c = 'red), label = 'loss)
# loss라는 값을 볼 것이고 '.'으로 어떠한 표시를 하고, 색깔은 red, loss라는 라벨을 붙인다
# marker와 color에는 몇 가지 종류가 있다, 본인이 꾸미고 싶은 대로 꾸미면 될 듯? 
# 북마크에 어떤 종류들이 있는지 확인할 수 있는 사이트 넣어둠
# plt.grid() 으로 그래프에 모눈(?)을 표시할 수 있다
# plt.title('~~')로 그래프의 제목 설정 가능
# plt.ylabel('loss)')로 y축 설정, plt.xlabel('epoch')로 x축 설정
# epoch에 따른 loss 값의 변화를 볼 수 있다
# plt.legend(loc = 'upper right') 그래프의 우측 상단에 각 선에 대한 정보가 담긴 상자가 생성된다

