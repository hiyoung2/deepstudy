# --title--
# 2742번: 기찍 N

# --problem_description--
# 자연수 N이 주어졌을 때, N부터 1까지 한 줄에 하나씩 출력하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄에 100,000보다 작거나 같은 자연수 N이 주어진다.

# --problem_output--
# 첫째 줄부터 N번째 줄 까지 차례대로 출력한다.

# 1. 첫째 줄에 n 입력 받기(input 또는 sys.stdin.readline() 사용)
# 2. 하나씩 차례대로 출력 -> for 문 사용
# 3. for문이 실행되면 n에서 1씩 감소하면서 하나씩 출력되어야 한다(0은 X)
# - 1번 실행시 n
# - 2번 실행시 n-1
# - 3번 실행시 n-2 ...

n = int(input()) #1
for i in range(n) : #2
    print(n)
    n -= 1 #3 : 출력을 먼저 하고 for문으로 올라가기 전에 1을 빼 주는 조건을 만들어둔다