##### 수정 필요
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error


train = pd.read_csv("./data/dacon/comp1/train.csv", header = 0, index_col = 0)

test = pd.read_csv("./data/dacon/comp1/test.csv", header = 0, index_col = 0)

submission = pd.read_csv("./data/dacon/comp1/sample_submission.csv", header = 0, index_col = 0)

print("train.shape : ", train.shape)             # (10000, 75) : x_train, x_test로 만들어야 함
print("test.shape : ", test.shape)               # (10000, 71) : x_pred
print("submission.shape : ", submission.shape)   # (10000, 4)  : y_pred

# 결측치

print(train.isnull().sum()) 

train = train.interpolate() 

test = test.interpolate()

# print(train.head())
train = train.fillna(method = 'bfill')
print(train.head())

train = train.values
test = test.values
submission = submission.values

print(type(train))

# 1. 데이터

# npy 형식으로 저장
np.save("./data/dacon/comp1/train.npy", arr = train)
np.save("./data/dacon/comp1/test.npy", arr = test)
np.save("./data/dacon/comp1/submission.npy", arr = submission)

# 데이터 불러오기
data = np.load("./data/dacon/comp1/train.npy",  allow_pickle = True)
x_pred = np.load("./data/dacon/comp1/test.npy", allow_pickle = True)
y_pred = np.load("./data/dacon/comp1/submission.npy", allow_pickle = True)

print("data.shape :", data.shape)
print("x_pred.shape :", x_pred.shape)
print("y_pred.shape :", y_pred.shape)

x = data[:, :71]
y = data[:, -4:]

print("x.shape :", x.shape)  # (10000, 71)
print("y.shape :", y.shape)  # (10000, 4)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 11
)

scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 구성
parameters = {
    "n_estimators" : [10, 20, 30, 100], "max_depth" : [4, 8, 10, 12, 20], 
    "min_samples_leaf" : [3, 5, 7, 9], "min_samples_split" : [3, 5, 7, 9]}

kfold = KFold(n_splits = 5, shuffle = True) 

model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv = kfold, random_state = 2)


# 3. 훈련
model.fit(x_train, y_train)

'''
# 4. 평가, 예측
y_pred = model.predict(x_pred)


print("y_pred :", y_pred)

print("최적의 매개변수 : ", model.best_estimator_)

mae = mean_absolute_error(y_pred, y_test)

print("mae :", mae)

# CSV 파일로 최종 변환
from pandas import DataFrame
a = np.arange(10000,20000)
submission = y_pred
submission = pd.DataFrame(submission,a)
submission.to_csv("./data/dacon/comp1/sample_submission.csv", header = ["hhb", "hbo2", "ca", "na"], index = True, index_label="id" )
'''