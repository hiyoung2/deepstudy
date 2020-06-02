# from keras.preprocessing import OneHotEncoder

# onehotencoder는 항상 해야 하는 것 , 분류모델에서


# from sklearn.preprocessing import OneHotEncoder # one-hot encoder 싸이킷런에 있음
# aaa = OneHotEncoder()
# aaa.fit(y)
# y = aaa.transform(y).toarray()

# 원-핫 인코딩(one-hot encoding)

# 컴퓨터 또는 기계는 문자보다는 숫자를 더 잘 처리할 수 있다
# 이를 위해 자연어 처리에서는 문자를 숫자로 바꾸는 여러가지 기법들이 있다
# 원-핫 인코딩은 그 많은 기법들 중 단어를 표현하는 가장 기본적인 표현 방법!
# 머신러닝, 딥러닝을 하기 위해서는 반드시 배워야 하는 표현 방법이다

# 원-핫 인코딩에 대해 배우기 전, 단어집합(vocabulary)에 대해 정의해보자
# '사전'이라고도 부르지만 '집합'이라는 표현이 보다 명확?
# 단어 집합은 앞으로 자연어 처리에서 계속 나오는 개념

# 단어 집합에서는 기본적으로 book과 books와 같이 단어의 변형 형태도 다른 단어로 간주한다
# 이 책에서는 앞으로 단어 집합에 있는 단어들을 가지고 문자를 숫자(더 구체적으로는 벡터)로 바꾸는 원-핫 인코딩을 포함한 여러 방법에 대해 배운다

# 원-핫 인코딩을 위해서 먼저 해야할 일은 단어 집합을 만드는 일이다
# 텍스트의 모든 단어를 중복을 허용하지 않고 모아놓으면 이를 단어 집합이라고 한다
# 그리고 이 단어 집합에 고유한 숫자를 부여하는 정수 인코딩을 진행한다
# 텍스트에 있는 단어가 총 5,000개라면 단어 집합의 크기는 5,000이다
# 5,000개의 단어가 있는 이 단어 집합의 단어들마다 1부터 5,000번까지 인덱스를 부여한다고 생각하자
# 가령 book은 150번, dog는 171번, love는 192번, books는 212번과 같이 부여할 수 있다

# 1. 원-핫 이코딩이란?
# 원-핫 인코딩은 단어 집합의 크기를 벡터의 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고, 
# 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식이다 
# 이렇게 표현된 벡터를 원-핫 벡터(One-Hot Vector)라고 한다

# 원-핫 인코딩을 두 가지 과정으로 정리해보자
# (1) 각 단어에 고유한 인덱스를 부여한다(정수 인코딩)
# (2) 표현하고 싶은 단어의 인덱스의 위치에 1을 부여하고, 다른 단어의 인덱스의 위치에는 0을 부여한다

########################################
# a = model.predict([1,2,3,4,5])
# print(np.argmax(a, axis = 1)+1)

# 다중분류 모델에서 데이터 전처리에 꼭 필요한 one_hot_ecoding
# one-hot-encoding이란 단 하나의 값만 True, 나머지는 모두 False인 인코딩
# 즉, 1개만 Hot(True)이고 나머지는 Cold(False)이다
# 예를 들면 [0,0,0,0,1]이다
# 5번째(zero-based index이니까 4)만 1이고 나머지는 0이다
# 데이터를 0 아니면 1 로 짝을 맞춰주기 위해서
# from keras.utils import np_utils
# y = np_utils.to_categorical(y)
# 를 써 주거나
# 
# from sklearn.preprocessing import OneHotEncoder # one-hot encoder 싸이킷런에 있음
# aaa = OneHotEncoder()
# aaa.fit(y)
# y = aaa.transform(y).toarray()
# 를 써줘야 하는데

# 차이점은 np_utils.to_categorical의 경우
# y data가 0부터 시작하지 않으면 슬라이싱 등을 통해 index를 조절해줘야 한다

# 반면 one-hot encoder는 0부터 시작하지 않더라도 알아서 데이터 shape에서 첫 번째 열에 0을 없애주고 
# 모델에 넣을 수 있는 형태로 만들어주는데, 이 때 중요한 것은 차원을 맞춰줘야 한다는 거다
# 예를 들면
# y = np.array([1,2,3,4,5,1,2,3,4,5])
# print(y.shape) # (10, ) 현재 1차원 vector 형태
# # one-hot encoder 는 2차원 형태로 넣어줘야 함
# # y = y.reshape(-1, 1) # -1? : 제일 끝, 
# == y = y.reshape(10, 1) # -1과 10 같음
# 2차원으로 변형!