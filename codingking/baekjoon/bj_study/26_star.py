# --url--
# https://www.acmicpc.net/problem/2523

# --title--
# 2523번: 별 찍기 - 13

# --problem_description--
# 예제를 보고 규칙을 유추한 뒤에 별을 찍어 보세요.

# --problem_input--
# 3

# --problem_output--
# *
# **
# ***
# **
# *

# 1. 숫자 하나를(n) 입력받는다
# 2. 별이 찍히는 횟수는 2n - 1
# 3. for문을 사용
# - 입력받은 숫자 n번째까지는 n만큼의 별을 출력
# - n번째 이상(ex, n = 3이면 4번째)부터는 별을 n개에서 하나씩 줄어들게 출력
# - 두 가지 조건이 있으므로 if문 사용
# - 첫 번째 조건은 for문 실행 횟수가 n보다 작거나 같을 때
# - 두 번째 조건은 for문 실행 횟수가 n보다 클 때
# - n이 3이라면 4번째부터는 2개가 출력
# - i(i+1로 설정해둠)와 n을 이용하여 식을 만들면
# - 2 == 3-(4-3) 즉, n-(i-n) 식을 적용하면 된다 

# 시도 - 런타임 에러
# import sys
# n = int(sys.stdiin.readline())

# for i in range(2*n -1) :
#     i += 1
#     if i <= n :
#         print("*" * i)
#     else :
#         print("*" * (n-i))
# 오타 투성이였는데 런타임 에러가 떠서 답은 맞는 줄 알고 계속 뻘짓

# 통과
import sys
n = int(sys.stdin.readline())

for i in range(2*n -1) :
    i += 1 # 출력되는 별의 개수를 정하려면 i가 0이 아닌 1부터 시작해야하므로
    if i <= n :
        print("*" * i) # 입력된 숫자 n번째까지는 i개만큼 출력
    else :
        print("*" * (n-(i-n))) # n+1 번째까지는 n에서 (i-n)을 뺀 만큼의 개수를 출력해야하므로