
# --url--
# https://www.acmicpc.net/problem/2953

# --title--
# 2953번: 나는 요리사다

# --problem_description--
# 	"나는 요리사다"는 다섯 참가자들이 서로의 요리 실력을 뽐내는 티비 프로이다. 각 참가자는 자신있는 음식을 하나씩 만들어오고, 서로 다른 사람의 음식을 점수로 평가해준다. 
# 점수는 1점부터 5점까지 있다.
# 	각 참가자가 얻은 점수는 다른 사람이 평가해 준 점수의 합이다. 이 쇼의 우승자는 가장 많은 점수를 얻은 사람이 된다.
# 	각 참가자가 얻은 평가 점수가 주어졌을 때, 우승자와 그의 점수를 구하는 프로그램을 작성하시오.

# --problem_input--
# 	총 다섯 개 줄에 각 참가자가 얻은 네 개의 평가 점수가 공백으로 구분되어 주어진다. 첫 번째 참가자부터 다섯 번째 참가자까지 순서대로 주어진다. 
# 항상 우승자가 유일한 경우만 입력으로 주어진다.

# --problem_output--
# 	첫째 줄에 우승자의 번호와 그가 얻은 점수를 출력한다.




import sys
count = 5
total = []

for i in range(count): 
    score = list(map(int, sys.stdin.readline().split())) # 다섯 사람의 점수를 리스트로 입력 받는다
    total_score = sum(score) # 각 다섯 사람의 점수 총합

    total.append(total_score) # 다섯 명의 점수 총합을 하나의 리스트로 둔다
    # print(total)

max_score = max(total) # 다섯 명의 점수의 최댓값을 max_score 변수에 대입
order = total.index(max(total)) + 1 # 가장 최고 점수가 나온 사람의 번호(순서)를 order에 대입
                                    # 총점들의 리스트의 index로 뽑아내는데 index는 0부터 시작하므로 1을 더해준다

print(order, max_score) # 우승자의 번호와 우승점수 출력



    
# 1. 일단 5명의 점수를 입력 받아야하는데 for문으로 횟수를 지정
# 2. 받은 점수들을 일단 리스트로 받아서 합계를 구한다
# 3. 각 참가자들의 합계를 한 리스트로 묶으면 그 중에서 최고 점수와 순서를 뽑아낼 수 있을 것 같았음

