# 인공지능의 겨울을 초래했던 XOR 문제

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1. 데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 0, 1]

# 서로 달라야 참이 되는 xor 형태의 데이터
# 달라야~ 참이 되는! xor 인데~ 왜!! 우리는 난제가 되었나~ 바람이 분다, 겨울이 온다, 아아아, 인공지능의 겨울~
# 머신러닝으로 해결이 불가능하다고 여겨졌다, 꽤 오랫동안
# 그러나 누군가가 또 해결을 했다
# 먼저, 가장 기본적으로 썼던 LinearSVC를 쓰면 어떻게 될 것인지 실행해보자

# 2. 모델
model = LinearSVC() # ()를 잊지 마시오

# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가, 예측
x_test = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_pred = model.predict(x_test)

acc = accuracy_score([0, 1, 1, 0], y_pred)
print(x_test, "의 예측 결과 : ", y_pred)

print("acc : ", acc)

# acc :  0.5 만 주구장창
# acc가 0.5인 건 뭐다?
# 예측 할 가치가 없다는 것
# 모 아니면 도! 기다 아니다! 이걸로 뭘 할 수 있겠는가
# how to solve "xor problem" -> 다음 파일로 가시오