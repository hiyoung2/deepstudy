
# --url--
# https://www.acmicpc.net/problem/19532

# --title--
# 19532번: 수학은 비대면강의입니다

# --problem_description--
# 수현이는 4차 산업혁명 시대에 살고 있는 중학생이다. 코로나 19로 인해, 수현이는 버추얼 학교로 버추얼 출석해 버추얼 강의를 듣고 있다. 
# 수현이의 버추얼 선생님은 문자가 2개인 연립방정식을 해결하는 방법에 대해 강의하고, 다음과 같은 문제를 숙제로 냈다.

# 4차 산업혁명 시대에 숙제나 하고 앉아있는 것보다 버추얼 친구들을 만나러 가는 게 더 가치있는 일이라고 생각했던 수현이는 이런 연립방정식을 풀 시간이 없었다. 
# 다행히도, 버추얼 강의의 숙제 제출은 인터넷 창의 빈 칸에 수들을 입력하는 식이다. 각 칸에는 $-999$ 이상 $999$ 이하의 정수만 입력할 수 있다. 
# 수현이가 버추얼 친구들을 만나러 버추얼 세계로 떠날 수 있게 도와주자.

# --problem_input--
# 정수 $a$, $b$, $c$, $d$, $e$, $f$가 공백으로 구분되어 차례대로 주어진다. ($-999 \leq a,b,c,d,e,f \leq 999$)

# 문제에서 언급한 방정식을 만족하는 $\left(x,y\right)$가 유일하게 존재하고, 이 때 $x$와 $y$가 각각 $-999$ 이상 $999$ 이하의 정수인 경우만 입력으로 주어짐이 보장된다.

# --problem_output--
# 문제의 답인 $x$와 $y$를 공백으로 구분해 출력한다.

# x, y의 값을 구하려면 
# 각 방정식에서 x의 계수인 a와 d(또는 y의 계수 b와 e)를 같게 만들어준 후 두 방정식 간 뺄셈을 해 주면 x, y를 구할 수 있음
# 1번째 식에는 2번째 식의 x 계수를 곱해주고 2번째 식에는 1번째 식의 x 계수를 곱해주면 계수가 같아지고
# 그 상태에서 방정식끼리의 뺄셈을 통해 x y를 구해볼 수 있을 것?

# 1차시도 - 런타임 에러
# import sys
# a, b, c, d, e, f = map(int, sys.stdin.readline().split())

# # print(a, b, c, e, d, f)

# ad = a*d
# bd = b*d
# cd = c*d

# da = d*a
# ea = e*a
# fa = f*a

# # ad = da
# y = (cd-fa)//(bd-ea)

# x = (c - (b*y))//a

# print(x, y)

# 2차 시도 - 런타임 에러
# import sys
# a, b, c, d, e, f = map(int, sys.stdin.readline().split())

# y = (c*d-f*a)//(b*d-e*a)
# x = (c - (b*y))//a

# print(x, y)

# a*d를 곱하는 방법 말고 a, d의 최소공배수 구하는 방법으로 접근
# '파이썬 최소공배수' 구글링 -> 최돼공약수를 구해주는 math 모듈의 gcd 함수가 존재
# gcd 함수를 통해 최소공배수를 쉽게 구할 수 있다

# 최대공약수
# from math import gcd
# print(gcd(12, 48)) # 12

# 최소공배수
# 최소공배수는 x와 y의 공통된 배수 가운데 최솟값을 의미한다, 더 쉽게 말해서
# 최소공배수는 주어진 수인 x, y의 곱에서 x, y의 최대 공약수를 나누어준 것과 같다
# from math import gcd
# def lcm(x, y) :
#     return x * y // gcd(x, y) # gcd(x, y) : x, y의 최대 공약수

# print(lcm(12, 48)) # 48


# 3차시도 - 런타임에러
# import sys
# a, b, c, d, e, f = map(int, sys.stdin.readline().split())

# from math import gcd

# def lcm(num1, num2) :
#     return num1 * num2 // gcd(num1, num2)

# # print(lcm(a, d))
# n = lcm(a, d) # n : a, d의 최소공배수

# y = ((n/a)*c -  (n/d) * f)/((n/a) * b - (n/d) * e)
# x = (c - b*y) / a

# x = int(x)
# y = int(y)

# print(x, y)

# 구글링 : 연립방정식 파이썬
# 연립방정식을 행렬로 접근하면 된다, 연립방정식은 행렬로 표현할 수 있으므로
# ex)
# (a b) (x)  =  (c) 
# (d e) (y)     (f)
# ax + by = c
# dx + ey = f
# 파이썬의 linalg.solve() 함수 : 연립방정식을 풀어준다
# numpy 모듈에 포함되어 있음

# 4차시도 - 런타임에러
# import sys 
# import numpy as np
# a, b, c, d, e, f = map(int, sys.stdin.readline().split())

# A = np.array([[a, b], [d, e]])
# # print(A)
# B = np.array([c, f])
# C = np.linalg.solve(A, B)
# # print(C) # [2. -1.]
# # print(C[0]) # 2.0

# x = int(C[0])
# y = int(C[1])

# print(x, y)

import sys 
import numpy as np
a, b, c, d, e, f = map(int, sys.stdin.readline().split())


A = [[a, b], [d, e]]
B = [c, f]

result = np.linalg.solve(A, B)

x = result[0]
y = result[1]

print(x, y)


