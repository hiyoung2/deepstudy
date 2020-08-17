# import library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA

import time
import random
import datetime

# load data

train = pd.read_csv("./dacon/dacon_emnist/data/train.csv")
test = pd.read_csv("./dacon/dacon_emnist/data/test.csv")
submission = pd.read_csv("./dacon/dacon_emnist/data/submission.csv")

# data 구성
# 본 문제의 목표는 기존의 mnist와 다르게 문자 속에 숨어 있는 숫자를 예측하는 것

print("train_haed")
print(train.set_index('id').head()) # [5 rows x 786 columns] # (id=index) digit, letter, pixels(0~783) : 786
print()
print("train.sahpe :", train.shape) # (2048, 787)
print()
print("test_haed")
print(test.set_index('id').head()) # [5 rows x 785 columns] # (id=index) letter, pixels(0~783) : 785
print()
print("test.sahpe :", test.shape) # (20480, 786)
print()
print()
print("submission_head")
print(submission.set_index('id').head()) 
print()
print("submission.shape :", submission.shape) # (20480, 2) # id, digit-> digit를 예측해야 한다


'''
train_haed
    digit letter  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  ...  760  761  762  763  764  765  766  767  768  769  770  771  772  773  774  775  776  777  778  779  780  781  782  783
id                                                                                              ...
1       5      L  1  1  1  4  3  0  0  4  4  3   0   4   3   3   3   4   4   0   0   1   1   3  ...    0    1    0    0    3    0    0    4    2    0    3    4    1    1    2    1    0    1    2    4    4    4    3    4 
2       0      B  0  4  0  0  4  1  1  1  4  2   0   3   4   0   0   2   3   4   0   3   4   3  ...    3    3    1    2    4    4    4    2    2    4    4    0    4    2    0    3    0    1    4    1    4    2    1    2 
3       4      L  1  1  2  2  1  1  1  0  2  1   3   2   2   2   4   1   1   4   1   0   1   3  ...    2    0    4    4    1    3    0    3    2    0    2    3    0    2    3    3    3    0    2    0    3    0    2    2 
4       9      D  1  2  0  2  0  4  0  3  4  3   1   0   3   2   2   0   3   4   1   0   4   1  ...    3    4    3    0    4    1    2    4    1    4    0    1    0    4    3    3    2    0    1    4    0    0    1    1 
5       6      A  3  0  2  4  0  3  0  4  2  4   2   1   4   1   1   4   4   0   2   3   4   4  ...    2    1    0    3    2    2    2    2    1    4    2    1    2    1    4    4    3    2    1    3    4    3    1    2 

[5 rows x 786 columns]

train.sahpe : (2048, 787)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_haed
     letter  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  22  ...  760  761  762  763  764  765  766  767  768  769  770  771  772  773  774  775  776  777  778  779  780  781  782  783
id                                                                                             ...
2049      L  0  4  0  2  4  2  3  1  0  0   1   0   1   3   4   4   0   0   2   4   4   1   3  ...    1    2    4    0    2    1    2    4    1    1    3    2    1    0    2    0    4    2    2    4    3    4    1    4  
2050      C  4  1  4  0  1  1  0  2  2  1   0   3   0   1   1   4   1   2   0   2   2   0   4  ...    4    2    2    4    3    1    3    3    3    1    3    4    4    2    0    3    2    4    2    4    2    2    1    2  
2051      S  0  4  0  1  3  2  3  0  2  1   2   0   1   0   3   0   1   4   3   0   0   3   0  ...    3    1    1    4    1    2    4    0    0    0    0    2    3    2    1    3    2    0    3    2    3    0    1    4  
2052      K  2  1  3  3  3  4  3  0  0  2   3   2   3   4   4   4   0   1   4   2   2   0   1  ...    0    0    2    3    2    2    3    1    1    2    4    0    1    2    3    0    3    2    4    1    0    4    4    4  
2053      W  1  0  1  1  2  2  1  4  1  1   4   3   4   1   2   1   4   3   3   4   0   4   4  ...    4    3    4    3    0    1    0    1    1    2    1    1    0    2    4    3    1    4    0    2    1    2    3    4  

[5 rows x 785 columns]

test.sahpe : (20480, 786)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
submission_head
      digit
id
2049      0
2050      0
2051      0
2052      0
2053      0

submission.shape : (20480, 2)
'''

# 데이터 각 열 자료형 확인
print("train's dtype")
print(train.dtypes)
print()
print("test's dtype")
print(test.dtypes)
print()
print("submission's dtype")
print(submission.dtypes)

'''
train's dtype
id         int64
digit      int64
letter    object
0          int64
1          int64
           ...
779        int64
780        int64
781        int64
782        int64
783        int64
Length: 787, dtype: object

test's dtype
id         int64
letter    object
0          int64
1          int64
2          int64
           ...
779        int64
780        int64
781        int64
782        int64
783        int64
Length: 786, dtype: object

submission's dtype
id       int64
digit    int64
dtype: object

id        774
digit       4
letter      A
Name: 773, dtype: object
'''

# view image
img = train.query("letter == 'A'")[
    [(str(i)) for i in range(784)]
].iloc[28].values.reshape(28, 28)

plt.imshow(img)
plt.title(train.query("letter == 'A'").iloc[28]['digit'], fontsize=15)
# plt.show()

print()
print(train.query("letter == 'A'").iloc[28].iloc[:3])

'''
id        774
digit       4
letter      A
Name: 773, dtype: object
'''
# 위의 이미지는 A 문자 속에 4가 숨어 있음, 모든 이미지들은 위와 같이 생겼고, '4'를 예측하는 것이 목표!

################################################################################################################
# query 메서드 
# pandas에서 DataFrame은 query 메서드를 지원한다 # pd.DataFrame.query
# 이 메서드는 조건식을 문자열로 입력받아 해당 조건에 만족하는 행을 추출해 출력해주는 함수이다
# 사용 방법은 단순히 대괄호[]에서 조건식을 입력했던 것과 동일하게 입력을 해주면 되나, 차이점은 문자열이 들어간다는 것이다

# how to use query() method
# DataFrame.query(expr, inplace=False)
# expr : 입력되는 조건식
# inplace가 True일 경우 query에 의해 출력된 데이터로 원본 데이터를 대체한다
# inplace의 default == false

# 참고사항
# query 메서드를 굳이 사용하는 이유는?
# 사실 대괄호[]를 사용하여 조건식을 입력해도 되는데 굳이 query method를 사용하는 이유는
# 방대한 양의 데이터를 처리할 경우 이 메서드가 성능면에서 우위를 보여주기 때문
# pandas 공식 사이트에서는 약 200,000개의 행 이상을 처리할 경우에는 이 메서드가 성능면에서 우위를 보인다고 설명
# 하지만 적은 양의 데이터를 처리할 경우 큰 차이는 보이지 않는다

# query 메서드의 특징
# - 표현식을 문자열로 입력받는다
# - 단순하게 열의 레이블을 표현식에 넣어 사용이 가능하다
# - index 역시 표현식에 넣어 사용이 가능하다
# - 대용량의 데이터를 처리할 경우 대괄호 조건식을 사용하는 것보다 성능이 좋다
# - 비교연산자와 논리연산자 중 비교연산자가 우선순위가 있으므로 표현식이 단순해진다

# query에 적용되는 조건식의 문법 특징
# query 메서드와 대괄호 []를 사용한 결과는 동일하다

# 조건식 사용의 예
# DataFrame[ (DataFrame.a<DataFrame.b) & (DataFrame.b<DataFrane.c)]
# DataFrame.query('(a<b)&(b<c)')

# 하지만 비교연산자가 논리연산자보다 우위에 있다는 점을 이용하면 이런 문법을 더욱 단순화 시킬 수 있다
# 먼저 위 예제에서 소괄호()를 지울 수 있다
# 논리연산자의 경우 심볼 대신 영어를 사용할 수도 있다
# 비교연산자가 논리연산자(&과 |)보다 우선순위가 높다

# 아래 예제 (1), (2), (3)은 모두 동등한 결과를 보여준다

# (1) DataFrame.query('a<b & b<c')
# (2) DataFrame.query('a<b and b<c')
# (3) DataFrame.query('a<b<c')


# query 내용 보충 필요 - 사이트 참고 https://kongdols-room.tistory.com/120
##################################################################################################


# CNN 사용
# 아래 커널을 이미지마다 적용, Convolution 연산(가중합)을 취해 이미지의 특성을 파악하는 것이 목적

from scipy.signal import correlate2d

kernel = np.array([
    [0, -100, 0],
    [0, 255, 0],
    [0, -100, 0],
])

plt.imshow(correlate2d(img, kernel, mode='same'))
# plt.show()

# Baseline 구축
# LightGBM

# 문자 데이터를 one-hot encoding 하고
# 이미지 픽셀 데이터를 784개의 위치 feature라고 생각하고 concat

x_train = pd.concat(
    (pd.get_dummies(train.letter), train[[str(i) for i in range(784)]]), 
    axis=1)
y_train = train['digit']

# pd.get_dummies? 
# pandas.get_dummies() : One-Hot Encoding
# 머신러닝에서 문자로 된 데이터는 모델링이 되지 않는 경우가 있다
# 대표적으로 회귀분석은 숫자로 이루어진 데이터만 입력을 해야 한다
# 문자를 숫자로 바꿔주는 방법 중 하나로 One-Hot Encoding이 있다
# 가변수(dummy variable)로 만들어주는 것인데, 이는 0과 1로 이루어진 열을 나타낸다
# 1은 있다, 0은 없다를 나타낸다

print(x_train.head()) # [5 rows x 810 columns]
print()
print(y_train) # Name: digit, Length: 2048, dtype: int64


# train set을 8:2로 분리
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, train_size = 0.8, random_state = 77
)

from lightgbm import LGBMClassifier

lgb = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,
               importance_type='split', learning_rate=0.1,
               max_depth=9, min_child_samples=20, min_child_weight=0.001,
               min_split_gain=0.0, n_estimators=1000, n_jobs=-1, num_leaves=511,
               objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,
               silent=True, subsample=1.0, subsample_for_bin=200000,
               subsample_freq=0)

lgb.fit(x_train, y_train)

print((lgb.predict(x_valid) == y_valid.values).sum() / len(y_valid))

x_test = pd.concat(
    (pd.get_dummies(test.letter), test[[str(i) for i in range(784)]]), axis=1)

submission.digit = lgb.predict(x_test)

submission.to_csv('./dacon/dacon_emnist/submit/submission_0817_2.csv', index=False)

# 0.05 : 0.5292682926829269
# 0.06 : 0.551219512195122
# 0.07 : 0.5487804878048781
# 0.08 : 0.5487804878048781
# 0.09 : 0.5463414634146342
# 0.1 : 0.551219512195122


'''
lgb = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,
               importance_type='split', learning_rate=0.1,
               max_depth=9, min_child_samples=20, min_child_weight=0.001,
               min_split_gain=0.0, n_estimators=1000, n_jobs=-1, num_leaves=511,
               objective=None, random_state=None, reg_alpha=0.0, reg_lambda=0.0,
               silent=True, subsample=1.0, subsample_for_bin=200000,
               subsample_freq=0)
# 0.5804878048780487

'''