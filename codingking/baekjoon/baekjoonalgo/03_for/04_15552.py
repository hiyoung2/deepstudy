# --title--
# 15552번: 빠른 A+B

# --problem_description--
# 본격적으로 for문 문제를 풀기 전에 주의해야 할 점이 있다. 입출력 방식이 느리면 여러 줄을 입력받거나 출력할 때 시간초과가 날 수 있다는 점이다.

# 또한 입력과 출력 스트림은 별개이므로, 테스트케이스를 전부 입력받아서 저장한 뒤 전부 출력할 필요는 없다. 테스트케이스를 하나 받은 뒤 하나 출력해도 된다.

# --problem_input--
# 첫 줄에 테스트케이스의 개수 T가 주어진다. T는 최대 1,000,000이다. 다음 T줄에는 각각 두 정수 A와 B가 주어진다. A와 B는 1 이상, 1,000 이하이다.


# --problem_output--
# 각 테스트케이스마다 A+B를 한 줄에 하나씩 순서대로 출력한다.


import sys

cnt = int(input())
res = []

for i in range(cnt) :
    a, b = map(int, sys.stdin.readline().split())
    cnt -= 1
    total = a + b
    res.append(total)

for j in range(len(res)) :
    print(res[j])



