# winequality-white.csv 파일을 가지고 머신러닝 기법으로 모델을 짜 보자

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1. 데이터 준비

wine = pd.read_csv('./data/csv/winequality-white.csv',
                   header = 0, sep = ';')
# deepstudy - csv 폴더 내의 winequality-white.csv 파일을 
# pandas로 읽어서 wine이라는 이름안에 넣어 둔다
# 데이터의 행 중 index[0]에 해당하는 것을 header로 잡고
# 데이터 컬럼을 구분하기 위해 사용된 ;(세미콜론)은 빼 준 채로 wine 데이터를 생성
# idnex는 따로 지정해줄 필요가 없다
# 모든 컬럼들이 실질 데이터 자체들이라서 첫 번째 행만 header = 0으로 header 처리!
# 한글이 없으므로 encoding을 따로 안 써줘도 된다

y = wine['quality']
# 지금 판다스의 데이터 프레임의 구조이기 때문에
# 넘파이처럼 슬라이싱 해 줄 필요 없이, y에 해당하는 데이터 컬럼의 이름을 이런 식으로 써 주면 된다

x = wine.drop('quality', axis = 1)
# x는 winde 데이터에서 열을 기준으로, 'quality'라는 데이터 컬럼을 drop(낙오, 탈락, 배제)하고
# 남은 컬럼들을 x 값으로 잡겠다

# 보충!
# pandas에서는 iloc, loc라는 개념으로 데이터 슬라이싱이 가능하다
# 
#
#
#


print("x.shape : ", x.shape) # (4898, 11)
print("y.shape : ", y.shape) # (4898,)

# 수업 시간에 이 데이터를 가지고 배운 대로 머신러닝 그리고 케라스로 모델을 구성했었는데
# acc가 잘 나오지 않았는데, 그 이유는 y 데이터 값들에 문제가 있었다
# 데이터를 살펴보면, 대충 봐도 y에 해당하는 'quality'의 값들이 5, 6이 엄청 많다는 걸 알 수 있다
# 5, 6이 워낙 많은 부분을 차지하다 보니, 머신은 그냥 어떤 것들을 예측해도
# 5, 6이면 평타는 치겠지~ 라는 생각으로 5, 6에만 집중(?), 선택(?)을 하게 되는 상황이 벌어진다
# 따라서, 우리는 5, 6에만 치우쳐져 있는 y data를 골고루 반영할 수 있도록 무언가 조치를 취해줘야 한다
# 어떤 방식으로? y의 라벨들을 축소시켜준다!
# 먼저, 확실히 y의 값들이 어떻게 분포되어 있는지, 알아볼 수 있는 방법이 있다

wine.groupby('quality')['quality'].count()
# wine 데이터의 'quality' 컬럼들의 구성 요소들의 갯수를 알아볼 수 있게 한다
# 문법적인 것이니 그냥 이렇게 쓴다고 알고 있어야 할 듯

count_data = wine.groupby('quality')['quality'].count()
# count_data라는 변수에 count한 것들을 넣어 둔다

print(count_data)
# 그러면 갯수들을 모아 놓은 데이터를 출력해보자
'''
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64
이렇게 y의 'quality' 데이터들의 정보를 알 수 있다!
확실히 5와 6에 치우쳐져 있다는 것을 확인
'''

# 그림으로 더 와닿게 보자
# count_data.plot()
# plt.show()

# 확실히 치우쳐져 있는 것을 확인했으니, 얘네들을 어떻게 하면
# 결괏값에 지들만 영향 끼치는 일이 안 일어나게 할 수 있을지 생각해보자
# 현재 3~9등급까지 포진해있음
# 범위를 정해서 9가지를 3가지 정도로 줄여준다면?
# 골고루 분포하는 모습이 되지 않을까?
# y 레이블을 축소하는 것이 바로 그 개념
# for문을 통해 y 레이블을 축소해보자

newlist = [] # 먼저, 새로 레이블이 축소될 y data를 담을 빈 공간을 마련

print(type(y)) # <class 'pandas.core.series.Series'>
# print(list(y)) # y의 데이터들이 [], 리스트 형태로 출력된다

for i in list(y) :
    if i <= 4 :
        newlist += [0]
    elif i <= 7 :
        newlist += [1]
    else :
        newlist += [2]
y = newlist

# for문을 해석해보자
# i는 0부터 1, 2, 3, ... 시작
# list y 안의 숫자(i)가 4보다 작거나 같다면, 즉 quality가 4보다 작거나 같다면 
# 빈 리스트에 0이라는 값으로 넣어준다
# i가 4보다는 크고 7보다 작거나 같으면 빈 리스트에 1이라는 값으로 넣어준다
# i가 7보다 크다면!(앞의 두 조건이 모두 아니라면) 2라는 값으로 넣어준다

print("y_newlist : ", y)
# 0, 1, 2로 이루어진 데이터로 바뀌었다
# 등급이 총 3개로 바뀌었다 ( 3,4 / 5, 6, 7 / 8, 9)

# 1 - 1 데이터 전처리
x_train, x_test, y_train , y_test = train_test_split(
    x, y, train_size = 0.8
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)

scaler.fit(x_test)
x_test = scaler.transform(x_test)

# 2. 모델 구성

# model = SVC()

# model = KNeighborsClassifier()

model = RandomForestClassifier()

# 훈련 (머신러닝에서는 컴파일 과정이 없다)

model.fit(x_train, y_train)

# 평가, 에측
acc = model.score(x_test, y_test)
y_pred = model.predict(x_test)

print(y_pred)
print("acc       : ", acc)
print("acc_score : ", accuracy_score(y_test, y_pred))