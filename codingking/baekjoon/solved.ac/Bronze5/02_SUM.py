# 8393번 : 합

# 문제
# while문으로 일단 1부터 변수 n까지의 합을 구하는 프로그램 작성

# 통과
n = int(input())

result = 0
while n != 0 :
    result += n
    n -= 1
    if n == 0 :
        print(result)

# trial and error

# 합인데 팩토리얼 생각하고
# math 끌어와서 팩토리얼 계산하고 있었음

# 합이면 숫자 n이 주어졌을 때 
# n + (n-1) + (n-2)... 이렇게 계산 되어야 하니까
# n이 하나씩 줄어드는 작업이 반복되어야 한다는 느낌
# 반복문 while을 적용해야한다고 생각
# while 문의 형태
# while 조건식 : ...
# 조건식이 False가 될 때까지 처리를 반복

# 테스트 
# n = 10
# result = 0

# while n != 0 : # 0이 되면 더이상 처리문을 수행하지 않음
#     result += n
#     n -= 1
#     if n == 0 : # if로 조건을 달았음, n이 0이면 while문에서 수행한 result를 print하라
#         print(result)


