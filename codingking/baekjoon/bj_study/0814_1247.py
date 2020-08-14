
# --url--
# https://www.acmicpc.net/problem/1247

# --title--
# 1247번: 부호

# --problem_description--
# N개의 정수가 주어지면, 이 정수들의 합 S의 부호를 구하는 프로그램을 작성하시오.

# --problem_input--
# 총 3개의 테스트 셋이 주어진다. 각 테스트 셋의 첫째 줄에는 N(1≤N≤100,000)이 주어지고, 둘째 줄부터 N개의 줄에 걸쳐 각 정수가 주어진다. 주어지는 정수의 절댓값은 9223372036854775807보다 작거나 같다.

# --problem_output--
# 총 3개의 줄에 걸쳐 각 테스트 셋에 대해 N개의 정수들의 합 S의 부호를 출력한다. S=0이면 "0"을, S>0이면 "+"를, S<0이면 "-"를 출력하면 된다.

import sys
test = 3
s = 0

for i in range(test):
    i = int(sys.stdin.readline())
    s += i

if s == 0 :
    print("0")
elif s > 0 :
    print("+")
elif s < 0 :
    print("-")