'''
--url--
https://www.acmicpc.net/problem/10952

--title--
10952번: A+B - 5

--problem_description--
두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

--problem_input--
입력은 여러 개의 테스트 케이스로 이루어져 있다.

각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)

입력의 마지막에는 0 두 개가 들어온다.

--problem_output--
각 테스트 케이스마다 A+B를 출력한다.

'''

# 1. 두 정수를 한 줄에 입력 받는데, 0 0 이 입력되면 입력 중지
# 2. 마지막 0 0 을 제외한 입력된 숫자들의 합을 각각 출력
# 3. for? while? continue? break?

# # for i in range()
import sys
# num = []

# # while sum(num) != 0 :
# #     num = map, int(sys.stdin.readline().split())
# #     print(sum(num))

# for i in range(sum(num)==0) :
#     num = map(int, sys.stdin.readline().split())
#     print(sum(num))

num = []
while sum(num) != 0 :
    num = list(map(int, sys.stdin.readline().split()))
    num.append(num)
    print(num)