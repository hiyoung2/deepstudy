# --url--
# https://www.acmicpc.net/problem/11021

# --title--
# 11021번: A+B - 7

# --problem_description--
# 두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄에 테스트 케이스의 개수 T가 주어진다.

# 각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)

# --problem_output--
# 각 테스트 케이스마다 "Case #x: "를 출력한 다음, A+B를 출력한다. 테스트 케이스 번호는 1부터 시작한다.

import sys
cnt = int(sys.stdin.readline())
res = [] 
for i in range(cnt) :
    a, b = map(int, sys.stdin.readline().split()) 
    cnt -= 1 
    total = a + b 
    res.append(total) 

x = 0
for j in range(len(res)) : 
    x += 1
    print("Case #",x,": " + str(res[j]), sep = "")