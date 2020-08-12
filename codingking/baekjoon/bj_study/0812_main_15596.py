# --url--
# https://www.acmicpc.net/problem/15596

# --title--
# 15596번: 정수 N개의 합

# --problem_description--
# 정수 n개가 주어졌을 때, n개의 합을 구하는 함수를 작성하시오.

# 작성해야 하는 함수는 다음과 같다.

# --problem_input--

def solve(a: list):
    hap = sum(a)
    return hap

# test = [1, 2, 3, 4]
# print(solve(test))

# 문제 속에 답이 있었다 
# a: list

# 실패
# def solve(a):
#     a = list(map, int(input().split()))
#     return sum(a)

# print(solve())
    
# import sys

# a = list(map(int, sys.stdin.readline().split()))
# print(a)

# a = list(map(int, input().split()))
# print(a)

# def solve():
#     return map(int, input().split())
# solve()


