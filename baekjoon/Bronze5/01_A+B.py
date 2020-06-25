# 1000번 : 합

# 문제
# 두 정수 A와 B를 입력받은 다음, A + B를 출력하는 프로그램 작성

# 통과
a, b = map(int, input().split())
print(a + b)


# trial and error

# 1차 (원인 : 문제 제대로 파악 X)
# num1 = input('첫 번째 숫자 입력 :')
# num2 = input('두 번째 숫자 입력 :')

# num1 = int(num1)
# num2 = int(num2)

# print(num1 + num2)

# 조건이 있었음
# 첫 째 줄에 A + B 를 출력한다
# 예제 입력 1 2
# 예제 출력 3

# 한 줄에 두 가지 숫자를 입력받아야 함
# 아예 감이 잡히질 않아서 키워드로 검색
# '파이썬 input 여러 개' 로 구글링해서
# input에 여러 개의 값을 받을 수 있는 방법을 찾음

# 그런데 파이썬 딥러닝 교과서 인덱스에 split()이 있었다
# 인덱스에서 먼저 잘 살펴봐야겠음

# 1. 변수1, 변수2 = input().split()
# - 변수 2개를 받고 입력받은 값을 공백을 기준으로 분리해준다
# 2. 변수1, 변수2 = input().split('기준문자열')
# 3. 변수1, 변수2 = input('문자열').split()
# 4. 변수1, 변수2 = input('문자열').split('기준문자열')

# a, b = input("문자열 두 개를 입력하세요 : ").split()
# print(a)
# print(b)
# '''
# 문자열 두 개를 입력하세요: hello python
# hello
# python
# '''

# a, b = input(",를 사용해 문자열 두 개를 입력하세요 : ").split(',')
# print(a)
# print(b)
# '''
# ,를 사용해 문자열 두 개를 입력하세요 : 안,녕
# 안
# 녕
# '''

# input에 split을 사용하면 입력받은 값을 공백을 기준으로 분리하여 변수에 차례대로 저장한다

# 두 숫자의 합 구하기
# a, b = input('숫자 두 개를 입력하세요 : ').split()
# print(a + b)

# a에 1, b에 2를 넣으면 a + b 의 결과로 12가 나온다
# input에서 입력받은 값이 문자열, 이 문자열은 split을 해도 문자열이기 때문이다
# 따라서 입력 값을 정수로 변환해줘야 한다

# 내가 처음에 했던 방법
# a, b = input("숫자 두 개를 입력하세요 : ").split()
# a = int(a)
# b = int(b)

# print(a + b)

# split한 결과를 매번 int로 변환해줘야 해서 귀찮음, 더 짧고 편한 코드가 있다???
# 'map' 이라는 것을 사용하면 된다
# int, input("두 개의 숫자를 입력 : ").split()를 map으로 묶어주면
# split의 결과를 모두 int로 변환해준다! (실수로 변환하고 싶다면 float를 써 주면 되겠지)

# 변수1, 변수2 = map(int, input().split())
# 변수1, 변수2 = map(int, input().split('기준문자열'))
# 변수1, 변수2 = map(int, input('문자열').split())
# 변수1, 변수2 = map(int, input('문자열').split('기준문자열'))

# 변수는 값이나 계산 결과를 저장할 때 사용한다는 점, 변수를 만드는 방법, 변수 이름을 짓는 방법을 알고 있자
# 특히 input과 split의 결과가 문자열!이라는 점이 중요하다
# 따라서 숫자 계산을 해야한다면, int나 float을 사용하여 결과를 숫자로 변환해준다는 점을 잊지 말자
# split의 결과 모두를 int나 float로 변환해야하면 map을 사용하는 것이 편리하다!


# a, b = map(int, input("두 개의 숫자를 입력 : ").split())
# print(a + b)
# 틀렸다고 뜸, 입력하라는 문구 같은 거 임의대로 하면 안 되나 봄


# 한 번에 여러 개를 정수로 변환 시키기

# a = [1.1, 2.2, 3.3, 4.4, 5.5]

# # for문 사용
# for i in range(len(a)) :
#     a[i] = int(a[i])
# print(a)

# # map을 사용
# print(list(map(int, a)))