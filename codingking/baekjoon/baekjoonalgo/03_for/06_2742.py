# --url--
# https://www.acmicpc.net/problem/2742

# --title--
# 2742번: 기찍 N

# --problem_description--
# 자연수 N이 주어졌을 때, N부터 1까지 한 줄에 하나씩 출력하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄에 100,000보다 작거나 같은 자연수 N이 주어진다.

# --problem_output--
# 첫째 줄부터 N번째 줄 까지 차례대로 출력한다.

cnt = int(input())
for i in range(cnt) :
    print(cnt)
    cnt -= 1