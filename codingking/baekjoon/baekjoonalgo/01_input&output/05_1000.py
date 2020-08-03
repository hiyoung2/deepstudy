# --url--
# https://www.acmicpc.net/problem/1000

# --title--
# 1000번: A+B

# --problem_description--
# 두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄에 A와 B가 주어진다. (0 < A, B < 10)

# --problem_output--
# 첫째 줄에 A+B를 출력한다.

# 1
# a, b = map(int, input().split())
# sum = a+b
# print(sum)

# 2
import sys
c, d = map(int, sys.stdin.readline().split())
result = c + d
print(result)