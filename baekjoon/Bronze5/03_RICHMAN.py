# 1271번 : 엄청난 부자

# 문제 
# 예제 입력 : 1000 100 (금액(n), 인원(m))
# 예제 출력 
# 10 (인원당 받는 금액)
# 0 (남은 금액)
# 출력에서는 개행?

# 통과
n, m = map(int, input().split())

a, b = divmod(n, m)

print(a)
print(b)


# trial and error

# 런타임 에러가 나는 게 몫, 나머지 따로 따로 구하는 것 때문이라고 추측
# 몫, 나머지를 한 번에 도출 해 낼 수 있는 게 있을 것 같아서
# '몫과 나머지 파이썬 한 번에' 라는 키워드로 구글링
# 파이썬 내장 함수인 'divmod'를 찾음

# divmod는 2개의 숫자를 입력으로 받고 ex) divmod(a, b)
# a를 b로 나눈 몫과 나머지를 튜플 형태로 돌려주는 함수이다

# if, a = 10이고  b = 5일 때
# print(divmod(a,b)) -> (2, 0) 으로 출력 된다


# 1차 : 런타임 에러
# n, m = map(int, input().split())
# a = n / m
# b = n % m
# print(a)
# print(b)


# 2차 : 런타임 에러

# n, m = map(int, input().split())
# a, b = (n / m, n % m)
# print(a)
# print(b)




