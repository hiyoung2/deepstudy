'''
1. 회귀
2. 이진 분류
3. 다중 분류

1. eval에 'loss'와 다른 지표 1개 더 추가
2. earlyStopping 적용
3. plot으로 그릴 것

4. 결과는 주석으로 소스 하단에 표시
'''

# 그림을 그려 봅시다!
# 기본 코드로만 구성(나머지는 알아서 찾아보기)

from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

boston = load_boston()
x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)

model = XGBRegressor(n_estimators = 300, learning_rate = 0.1)

model.fit(x_train, y_train, verbose = True, eval_metric = ["logloss", "rmse"],
                            eval_set = [(x_train, y_train), (x_test, y_test)],
                            early_stopping_rounds = 10)

results = model.evals_result()
print("eval's results :", results)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)

print("R2 : %.2f%%" %(r2 * 100.0)) # R2 : 93.29%


# 그래프 그리기
epochs = len(results['validation_0']['logloss'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label = 'Train')
ax.plot(x_axis, results['validation_1']['logloss'], label = 'Test')

ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label = 'Train')
ax.plot(x_axis, results['validation_1']['rmse'], label = 'Test')

ax.legend()
plt.ylabel('RMSE')
plt.title('XGBoost RMSE')

plt.show()