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

# view image
img = train.query("letter == 'A'")[
    [(str(i)) for i in range(784)]
].iloc[28].values.reshape(28, 28)

plt.imshow(img)
plt.title(train.query("letter == 'A'").iloc[28]['digit'], fontsize=15)
plt.show()

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

