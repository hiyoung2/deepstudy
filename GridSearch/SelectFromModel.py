from sklearn.feature_selection import SelectFromModel
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

x, y = load_boston(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, shuffle = True, random_state = 66
)


model = XGBRegressor()

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print("R2 :", score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)


for thresh in thresholds: # 컬럼 수만큼 돈다! 빙글 빙글
               
    selection = SelectFromModel(model, threshold = thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    # print(select_x_train.shape)

    selection_model = XGBRegressor()
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)
    # print("R2 :", score)


    print("Trersh=%.3f, n = %d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
          score*100.0))