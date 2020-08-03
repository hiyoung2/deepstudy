# --url--
# https://www.acmicpc.net/problem/2741

# --title--
# 2741번: N 찍기

# --problem_description--
# 자연수 N이 주어졌을 때, 1부터 N까지 한 줄에 하나씩 출력하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄에 100,000보다 작거나 같은 자연수 N이 주어진다.

# --problem_output--
# 첫째 줄부터 N번째 줄 까지 차례대로 출력한다.

cnt = int(input())

for i in range(cnt) :
    print(i+1)