# --url--
# https://www.acmicpc.net/problem/10869

# --title--
# 10869번: 사칙연산

# --problem_description--
# 두 자연수 A와 B가 주어진다. 이때, A+B, A-B, A*B, A/B(몫), A%B(나머지)를 출력하는 프로그램을 작성하시오. 

# --problem_input--
# 두 자연수 A와 B가 주어진다. (1 ≤ A, B ≤ 10,000)

# --problem_output--
# 첫째 줄에 A+B, 둘째 줄에 A-B, 셋째 줄에 A*B, 넷째 줄에 A/B, 다섯째 줄에 A%B를 출력한다.

# 1
# a, b = map(int, input().split())

# sum = a + b
# sub = a - b
# mul = a * b
# quo = a // b
# mod = a % b

# print(sum)
# print(sub)
# print(mul)
# print(quo)
# print(mod)

# 2
import sys

a, b = map(int, sys.stdin.readline().split())

ans1 = a + b
ans2 = a - b
ans3 = a * b
ans4 = a // b
ans5 = a % b

answer = [ans1, ans2, ans3, ans4, ans5]

# print(answer)

for i in range(len(answer)) :
    print(answer[i])
    i += 1