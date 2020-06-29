# --title--
# 8393번: 합

# --problem_description--
# n이 주어졌을 때, 1부터 n까지 합을 구하는 프로그램을 작성하시오.


# --problem_input--
# 첫째 줄에 n (1 ≤ n ≤ 10,000)이 주어진다.


# --problem_output--
# 1부터 n까지 합을 출력한다.

# 1. n을 입력 받는다
# 2. 합계를 받을 변수를 0으로 설정
# 3. for 문으로 n까지의 수를 더하는 과정을 실행 
# - n까지 포함해서 계산해야하므로 range에는 n+1을 넣는다(n으로 넣으면 n을 포함하지 않기 때문)
# 4. 합계를 받을 변수에 i를 더해주면서 최종 출력 n까지의 합을 구한다

n = int(input()) # 1

total = 0 # 2

for i in range(n+1) : # 3
    total += i # 4

print(total)

