# Pipiline 사용, Meachin Learning 사용

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# 싸이킷런 모듈
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, accuracy_score, mean_absolute_error, mean_squared_error

# 케라스 모듈
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten
from keras.layers import Input

# dacon_comp1 필요한 데이터 불러오기(csv 파일 읽기)
# 항상 경로 주시하자
train = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)
test = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)
submit = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)
# pd.read_csv("./파일경로/경로/경로,,/파일명.csv", header 처리(현재는 0행을 header로 설정), index_col = 0(0번째 column을 index로 설정))
# header와 index_col은 파일 확인 후 바르게 설정 잘 해줘야 한다
# sep = ','은 안 적어줘도 된다 -> default! (디폴트가 참 많다,,)
# train / test / submit 은 내가 정하는 데이터 이름
# 식별하기 쉽게 잘 정하자

# shape 확인
print("train.shape :", train.shape)   # (10000, 75)
print("test.shape :", test.shape)     # (10000, 71)
print("submit.shape :", submit.shape) # (10000, 4)
print("===========================================")
# shape을 확인 해 보니까
# test의 column이 71, submit의 column이 4
# submit 파일을 확인 해 보면, 데이터가 모두 0 -> 우리가 제출해야 할 파일이라 데이터가 채워져 있지 않다
# 모델을 구성하여 우리는 submit을 y_pred로 잡아 예측 해야 한다
# 그러기 위해서는 x와 y가 모두 있는 train data를 훈련 시킬 하나의 dataset으로 잡아야 한다
# test 파일은 y가 없다 -> 즉 test를 y_pred를 예측하기 위한 x_pred로 잡아야 한다(x_pred : y_pred를 위한 x값 데이터, 이름은 마음대로 지어도 되나, 식별할 수 있게!)

# 정리 #
# train -> 3단계 컴파일, 훈련, 그리고 4단계의 평가에서 쓰일 dataset (train_test_split, 전처리가 필요)
# test -> 4단계 예측 단계에서 쓰일 dataset (y_pred = model.predict() 이 괄호 안에 들어갈 데이터)
# submit -> 우리가 예측해야 할 , 비어 있는 dataset으로 만든 모델로 예측 해야 하며 마지막에 csv파일로 변환하여 대회에 제출해야 한다

# 데이터 처리 - 결측치 확인 및 처리
print(train.isnull().sum())
'''
rho           0
650_src       0
660_src       0
670_src       0
680_src       0
           ...
990_dst    1987
hhb           0
hbo2          0
ca            0
na            0
Length: 75, dtype: int64
'''
# 현재 990_dst 라는 column에 NAN, 결측치가 1987개나 있다
# 결측치를 어떻게 처리할까?
# (1) 0으로 처리 -> 아무래도 문제가 생기겠지
# (2) fillna(method = 'bfill')(또는 'ffill')을 통해 바로 앞의 데이터값 또는 바로 뒤의 데이터값을 대입
# (3) fillna(데이터이름.mean())을 통해 평균값 대입
# (4) 보간법 사용 -> 그 중 '선형 보간'
# : 컬럼별로 데이터 값들의 선을 쭉 그으면 곳곳 비워져 있는 결측치의 자리를 그 선이 자연스럽게 이어질 수 있을 만한 값들로 채워준다
# 어떻게? 제공해주는 기능이니 그냥 쓰면 된다(평타 85점)
# 새로 배운 '선형 보간'을 써 보겠다
train = train.interpolate()
test = test.interpolate() 
# 그런데, 이 방법을 썼는데도 데이터들의 가장 첫 번째 행의 NAN 값들은 해결이 안 된다는 것을 발견
print(train.head()) # 가장 앞 부분 확인
print(test.head())
print(("==========================================="))
'''
train.head
0   ... 생략...          NaN  0.000000e+00 
test.head
10000   .. 생략 ..000000e+00          NaN  
''' 
# 선형보간법은 앞 뒤의 데이터들이 자연스럽게 선으로 이어질 수 있게끔 결측치를 채워주는 것인데
# 가장 앞 행의 데이터 값은 시작하는 지점으로 앞뒤 반영이 불가(뒤만 있으니까), 그래서 선형보간법으로 해결이 안 되는 것이라 생각됨
# 따라서 첫 행의 NAN 값은 기존에 사용했던 fillna method를 사용해서 채워주자
train = train.fillna(method = 'bfill')
test = test.fillna(method = 'bfill')
print(train.head())
print(test.head())

print(train.isnull().sum())
'''
rho        0
650_src    0
660_src    0
670_src    0
680_src    0
          ..
990_dst    0
hhb        0
hbo2       0
ca         0
na         0
'''
# 결측치가 모두 사라졌다

# 1. 데이터

# 1) npy 형식으로 저장하기
np.save("./dacon/comp1/npy/train.npy", arr = train)
np.save("./dacon/comp1/npy/test.npy", arr = test)

data = np.load("./dacon/comp1/npy/train.npy", allow_pickle = True)
x_pred = np.load("./dacon/comp1/npy/test.npy", allow_pickle = True)

# 2) 데이터 슬라이싱
# 현재 data 파일에는 x, y 모두 포함 되어 있는 data를 슬라이싱 
# x, y 데이터로 나누고 x_train, x_test, y_train, y_test로 분리 해 줘야 함
# x는 input data, y는 output data

x = data[:, :-4]
y = data[:, -4:]

# 슬라이싱 잘 되었는지 shape 확인
print("데이터 슬라이싱")
print("x.shape :", x.shape) # (10000, 71)
print("y.shape :", y.shape) # (10000, 4)

# 3) train_test_split
# fit 과정에 쓸 train set와 evluate (or score in ml)에 사용할 test set을 분리해줘야 한다
print("train_test_split")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 11, shuffle = True
)
print("x_train.shape :", x_train.shape)  # (8000, 71)
print("x_test.shape :", x_test.shape)    # (2000, 71)
print("y_train.shape :", y_train.shape)  # (8000, 4)
print("y_test.shape :", y_test.shape)    # (2000, 4)


# 2. 모델 구성
pipe = Pipeline([("scaler", StandardScaler()), ('ensemble', RandomForestRegressor())])
# 파이프라인 사용
# 파이프라인 사용법 : 변수명 = Pipiline([("scaler", 사용할 스케일러()), ('사용할모델모듈명', 사용할모델명())])

# 3. 훈련
pipe.fit(x_train, y_train)
# 케라스에서 model.fit 하는 것과 같다
# 머신러닝에서는 아주 간단하게 훈련 시킨다
# 현재 내가 사용하는 모델이 pipe라는 변수에 있으므로 pipe.fit(x_train, y_train) 이라고 쓴다
# 머신러닝은 컴파일 과정이 따로 없다

# 4. 평가, 예측
loss = pipe.score(x_test, y_test)
# x_test와 y_test로 loss를 구한다(이 부분은 딥러닝과 같음)
# 딥러닝 : loss = model.evaluate(x_test, y_test, batch_size = XXX)
# evluate 대신 머신러닝은 score를 쓴다!!!!!

y_pred = pipe.predict(x_test)
# 딥러닝에서는 평가지표 ex) mse, acc를 구할 때
# loss, mse = model.evaluate(x_test, y_test)로 구했지만
# 머신러닝에서는 따로 x_test를 모델로 돌렸을 때 예측값과
# y_test라는 실젯값을 평가지표에 넣어줘야 한다
# 그래서 먼저 x_test를 predict해서 y_pred라는 변수에 대입한다
# y_pred는 변수명으로 내가 식별할 수 있는 이름을 사용하면 된다

# 평가지표 mae
mae = mean_absolute_error(y_test, y_pred)
# y_test라는 준비된 실젯갑과 x_test로 예측한 y_pred 간의 차이를 알아보는 지표로 mae를 사용한다
# mse와 마찬가지로 낮을수록 성능이 좋은 모델

submit = pipe.predict(x_pred)
# 최종 x_pred를 사용해서 submit 할 예측값 설정
# y_pred라는 변수명을 이미 사용해서 어차피 최종적으로 낼 데이터이므로 submit 이라는 변수명을 사용

print("loss :", loss)
print("mae :", mae)
print("submit :", submit[:5, :])

# 최종 submit, csv 파일로 변환
# numpy로 변환 해서 header와 index가 없는 상황이므로 header, index를 함께 설정해주면서 최종 변환

a = np.arange(10000, 20000)
# 현재 index인 id가 10000부터 19999까지 있으므로 arange를 통해 위와 같이 설정

submit = pd.DataFrame(submit, a)
# pd.DataFrame(데이터명, 인덱스(현재 변수를 사용했으므로 변수명을 적어준다))
submit.to_csv("./data/dacon/comp1/submit1.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label = "id")
# to_csv를 사용하여 최종 파일을 저장할 경로, 파일명 지정
# header는 기존 제공되었던 sample_submission 파일을 참조하여 [] 안에 ""형식으로 기입
# index는 위에서 설정을 했으므로 index = True로 지정
# index_col은 "id" 도 적어줘야 한다