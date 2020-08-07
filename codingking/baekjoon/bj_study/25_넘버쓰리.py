'''
--url--
https://www.acmicpc.net/problem/10817

--title--
10817번: 세 수

--problem_description--
세 정수 A, B, C가 주어진다. 이때, 두 번째로 큰 정수를 출력하는 프로그램을 작성하시오. 

--problem_input--
첫째 줄에 세 정수 A, B, C가 공백으로 구분되어 주어진다. (1 ≤ A, B, C ≤ 100)

--problem_output--
두 번째로 큰 정수를 출력한다.

'''
# 행렬 덧셈 테스트 (이 파일이랑 상관 없음 -> 다른 곳으로 옮겨라)
# import numpy as np
# a = np.array([[1, 2, 3],[4, 5, 6]])
# b = np.array([10, 20, 30])
# c = np.array([[10, 20, 30]])


# print("a + b :", a+b)
# print("a + c :", a+c)


# 1. 세 수를 입력 받는다
# 2. 리스트로 입력 받아서 sort 같은 것 써서 큰 수 또는 작은 수 대로 정렬
# 3. 세 가지 숫자 중에 중간 인덱스를 출력하면 될 것 같다

import sys
num = list(map(int, sys.stdin.readline().split()))
num.sort()
# print(num)
print(num[1])