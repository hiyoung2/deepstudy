
# --url--
# https://www.acmicpc.net/problem/2775

# --title--
# 2775번: 부녀회장이 될테야

# --problem_description--
# 평소 반상회에 참석하는 것을 좋아하는 주희는 이번 기회에 부녀회장이 되고 싶어 각 층의 사람들을 불러 모아 반상회를 주최하려고 한다.

# 이 아파트에 거주를 하려면 조건이 있는데, “a층의 b호에 살려면 자신의 아래(a-1)층의 1호부터 b호까지 사람들의 수의 합만큼 사람들을 데려와 살아야 한다” 는 계약 조항을 꼭 지키고 들어와야 한다.

# 아파트에 비어있는 집은 없고 모든 거주민들이 이 계약 조건을 지키고 왔다고 가정했을 때, 주어지는 양의 정수 k와 n에 대해 k층에 n호에는 몇 명이 살고 있는지 출력하라. 
# 단, 아파트에는 0층부터 있고 각층에는 1호부터 있으며, 0층의 i호에는 i명이 산다.

# --problem_input--
# 첫 번째 줄에 Test case의 수 T가 주어진다. 그리고 각각의 케이스마다 입력으로 첫 번째 줄에 정수 k, 두 번째 줄에 정수 n이 주어진다. (1 <= k <= 14, 1 <= n <= 14)

# --problem_output--
# 각각의 Test case에 대해서 해당 집에 거주민 수를 출력하라.

########################################################
# 입력받은 Test case의 수 T에 따라 k, n 출력횟수가 정해진다
# 0층은 n호까지 있으면 1호실부터 n호까지 사람이 1명 2명 ,,, n명 이런 식으로 한명씩 증가 
# 1층 사람들은 0층의 사람 수에 의해 결정, 2층은 1층, 3층은 2층,,, 
# 일단 0층의 정보를 깔아둬야 할 것 같다

import sys

t = int(sys.stdin.readline())

print(t)

for i in range(t) :
    k = int(sys.stdin.readline())
    n = int(sys.stdin.readline())

    # print(k)
    # print(n)
    people = [] # 0층의 호별 사람 정보
    for p in range(1, n+1):
        people.append(p) # n이 3, 즉 3호까지 있다면 [1, 2, 3] 1호는 1명, 2호는 2명, 3호는 3명
    # print(people)

    for j in range(k): # 층수에 따라 for문 실행 / k층의 이전 층의 사람 정보가 누적되어야 하므로
        for m in range(1, n): # n-1번 for문 실행 / 각 층의 1호는 항상 1명 , 그걸 빼고 2호 이상의 사람 수 계산이 필요하므로
            people[m] += people[m-1] # k층의 1호는 0호실 사람을 +, 2호는 0호실 + 1호실,,, 
                                      
    print(people[n-1]) # index가 0부터 시작, 호는 1부터 시작 => ex) 3호는 people의 index2



#######################################################
'''
cnt = int(input())
p = []
for a in range(1, cnt+1):
    p.append(a)

# print(p) # [0, 1, 2] # range(cnt)
print(p) # [1, 2, 3] # range(1, cnt+1)
'''