# 1330 : 두 수 비교하기
# --url--
# https://www.acmicpc.net/problem/1330

# --title--
# 1330번: 두 수 비교하기

# --problem_description--
# 두 정수 A와 B가 주어졌을 때, A와 B를 비교하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄에 A와 B가 주어진다. A와 B는 공백 한 칸으로 구분되어져 있다.

# --problem_output--
# 첫째 줄에 다음 세 가지 중 하나를 출력한다.


# a, b = map(int, input().split())
import sys

a, b = map(int, sys.stdin.readline().split())

if a > b :
    print(">")
elif a < b :
    print("<")
else :
    print("==")