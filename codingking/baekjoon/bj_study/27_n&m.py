
# --url--
# https://www.acmicpc.net/problem/18883

# --title--
# 18883번: N M 찍기

# --problem_description--
# 자연수 N, M이 주어졌을 때, 1부터 N×M까지 출력 형식대로 출력해보자.

# --problem_input--
# 첫째 줄에 공백 한 칸으로 구분한 N, M이 주어진다. 두 수는 1,000보다 작거나 같은 자연수이다.

# --problem_output--
# 총 N개의 줄을 출력해야 한다. 각 줄에는 M개의 정수를 공백 한 칸으로 구분해 출력해야 한다. 
# 1번 줄에는 1부터 M까지, 2번 줄에는 M+1부터 2×M까지, ..., N번 줄에는 (N-1)×M+1부터 N×M까지 출력해야 한다.

'''
입력
3 4

출력
1 2 3 4 (1 ~ M)
5 6 7 8 (M+1 ~ 2*M)
9 10 11 12 ((N-1)*M+1 ~ N*M  )
'''

# 1. 두 개의 숫자 n, m을 입력 받는다
# 2. 일단 1부터 n*m까지 그냥 출력하는 것부터 시도해보고 (ok) 그 다음 줄바꿈 하는 것 시도
# - 줄바꿈을 하려면 if문으로 조건 줘서?

# import sys
# n, m = map(int, sys.stdin.readline().split())
# for i in range(n*m) :
#     print(i+1)

'''
3 4
1
2
3
4
5
6
7
8
9
10
11
12
'''
# 시도
# import sys
# n, m = map(int, sys.stdin.readline().split())
# end = n * m # 가장 마지막에 나올 숫자 end 변수에 대입
# tmp = []
# for i in range(end) :
#     i += 1
#     tmp.append(i)
# print(tmp[:m])
# print(tmp[m:2*m])

# 조건을 주면 줄바꿈 적용되어서 출력하게 해야하는데 쉽사리 해결이 안 돼서 구선생에게 S.O.S
# 구글링 : 파이썬 리스트 요소 줄바꿈
# k = 10
# result = list(range(50))
# for i in result :
#     if i%k == 0:
#         print()
#     print(i, end = ' ') # 파이썬에서 print 함수 인자 'end'에 원하는 문자를 넣어주면 줄바꿈 대신 원하는 문자가 추가되어 출력된다!

# n, m = map(int, sys.stdin.readline().split())
# cnt = n*m
# result = list(range(cnt))

# for i in result :
#     if result[i] % m == 0 :
#         print()

#     print(result[i], end = ' ')

# 0 1 2 3
# 4 5 6 7
# 8 9 10 11 -> 이렇게 출력됨;

# n, m = map(int, sys.stdin.readline().split())
# cnt = n * m
# result = []

# 문제 파악을 위해 result를 출력 해 봄
# print(result) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 리스트 수정

# for i in range(cnt) :
#     i += 1
#     result.append(i)

# print(result)
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#  0  1  2  3  4  5  6  7  8  9   10  11

# print(result[0] + 1)
# print(type(result))
# print(m)

# for j in range(cnt) :
#     if result[j] % m == 1 :
#         print()
#     print(result[j], end = ' ')


# 최종 : 통과
import sys
n, m = map(int, sys.stdin.readline().split())
cnt = n * m

for i in range(cnt) :
    i += 1
    if i % m == 0 :
        print(i)
        continue
    print(i, end = ' ')


# continue : 조건이 참이면 반복문 계속
# 조건이 참일 때, continue 아랫 부분의 code block을 실행하지 않고!
# 처음의 반복 부분으로 되돌아가서 반복 loop을 계속 하도록 한다
# 위의 문제로 예시를 들면
# i가 m의 배수일 때, if문이 참일 때 i가 출력되고
# continue 때문에 더이상의 코드는 실행하지 않고 다시 for문으로 돌아간다
# i가 m의 배수가 아닐 떄, if문이 거짓이면 
# 다음 print문이 실행되고, ' ' 공백을 가지고 있어야 하므로 
# end 인자를 사용해 공백을 포함하여 출력해준다
# 주의할 것은 m의 배수가 되는, 즉 한 줄의 마지막 끝자리 숫자 뒤에는 공백이 없어야함!