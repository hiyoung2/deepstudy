# xor 문제를 해결해보자

from sklearn.svm import LinearSVC, SVC 
# 타란, LinearSVC가 아닌 그냥 SVC라는 모델이 존재했다
# 싸이킷런의 support vector machine, svm에서 불러온다
# 서포트 벡터 머신(support vector machine, SVM.)은,,
# '기계 학습의 분야 중 하나로 패턴 인식, 자료 분석을 위한 지도 학습 모델이며, 주로 분류와 회귀 분석을 위해 사용한다'고 한다


from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 0, 1]

# 2. 모델
model = SVC()

# 3. 훈련
model.fit(x_data, y_data)