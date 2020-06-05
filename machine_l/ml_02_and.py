# 머신러닝의 기본

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 0, 0, 1]

# 딱~ 보니까 뭐? and 
# 0, 0 : 0, 1, 0 = 0, 0, 1 = 0, 1, 1 = 1

# 2. 모델
model = LinearSVC()

# linear , 선형 즉, 이 모델은 회귀분석을 위한 모델
# linear, regressor ~ 들은 '회귀'라고 보면 된다
# 그런데, logistic regressor는? 분류이다! (시험 면접에서 많이 물어봤던 질문)
# 분류모델은 classifier가 들어간 모델들
# 머신러닝의 모델 구성은 저 한 줄이 끝! so simple
# 레이어 구성 이런 거 다 필요 없음, 이미 만들어져 있기 때문에
# customize 할 것 들은,, 어떤 모델을 골라서 쓸 지, 그리고 아직 안 배웠지만 () 안에 넣을 것들만 정하는 게 전부이다
# 머신러닝보다 좀 더 복잡(?)한 케라스를 먼저 배웠기 때문에 머신러닝 쉽게 받아들일 수 있다(선생님의 큰 뜻,,)


# 3. 훈련
model.fit(x_data, y_data)
# 이게 끝! compile도 필요 없다, wow

# 4. 평가, 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_pred = model.predict(x_test) # predict 방식은 케라스와 동일

acc = accuracy_score([0, 0, 0, 1], y_pred) 
# 케라스로 따지면 loss, mse(or acc) = model.evaluate(x_test, y_pred) 과정
# 케라스에서 metrics가 mse(or acc), 
# 머신러닝에서는 metrics로 accuracy_score를 import 해서 사용한다

print(x_test, "의 예측 결과 : " ,y_pred)
print("acc : ", acc)


# 세상 간단한 모델이라 acc 1.0이 나올 수 밖에
# and 문제, 머신러닝으로 간편하게 해결 가능
'''
[[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측 결과 :  [0 0 0 1]
acc :  1.0
'''