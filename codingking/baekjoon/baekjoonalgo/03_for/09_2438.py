'''
--url--
https://www.acmicpc.net/problem/2438

--title--
2438번: 별 찍기 - 1

--problem_description--
첫째 줄에는 별 1개, 둘째 줄에는 별 2개, N번째 줄에는 별 N개를 찍는 문제

--problem_input--
첫째 줄에 N(1 ≤ N ≤ 100)이 주어진다.

--problem_output--
첫째 줄부터 N번째 줄까지 차례대로 별을 출력한다.

'''

cnt = int(input())
    
for i in range(cnt):
    print("*"*(i+1))