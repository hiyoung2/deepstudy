# --url--
# https://www.acmicpc.net/problem/1008

# --title--
# 1008번: A/B

# --problem_description--
# 두 정수 A와 B를 입력받은 다음, A/B를 출력하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄에 A와 B가 주어진다. (0 < A, B < 10)

# --problem_output--
# None

# 1
# a, b = map(int, input().split())
# div = a/b
# print(div)

# 2
import sys
a, b = map(int, sys.stdin.readline().split())
result = a / b
print(result)

