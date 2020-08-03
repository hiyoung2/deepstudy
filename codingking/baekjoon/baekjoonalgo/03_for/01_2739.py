# 2739 : 구구단
# --url--
# https://www.acmicpc.net/problem/2739

# --title--
# 2739번: 구구단

# --problem_description--
# N을 입력받은 뒤, 구구단 N단을 출력하는 프로그램을 작성하시오. 출력 형식에 맞춰서 출력하면 된다.

# --problem_input--
# 첫째 줄에 N이 주어진다. N은 1보다 크거나 같고, 9보다 작거나 같다.

# --problem_output--
# 출력형식과 같게 N*1부터 N*9까지 출력한다.


n = int(input())
for i in range(9) :
    i += 1
    print("{0} * {1} = {2}".format(n, i, n*i))
