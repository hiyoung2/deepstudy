# --url--
# https://www.acmicpc.net/problem/2588

# --title--
# 2588번: 곱셈

# --problem_description--

# 	(세 자리 수) × (세 자리 수)는 다음과 같은 과정을 통하여 이루어진다.

# None

# 	(1)과 (2)위치에 들어갈 세 자리 자연수가 주어질 때 (3), (4), (5), (6)위치에 들어갈 값을 구하는 프로그램을 작성하시오.

# --problem_input--

# 	첫째 줄에 (1)의 위치에 들어갈 세 자리 자연수가, 둘째 줄에 (2)의 위치에 들어갈 세자리 자연수가 주어진다.
# --problem_output--

# 	첫째 줄부터 넷째 줄까지 차례대로 (3), (4), (5), (6)에 들어갈 값을 출력한다.



first = int(input())

second = input()

second_3 = int(second[-1])
second_2 = int(second[-2])
second_1 = int(second[-3])

third = first*second_3
fourth = first*second_2
fifth =first*second_1

sixth = first * int(second)

print(third)
print(fourth)
print(fifth)
print(sixth)


# 간결한 코드
# A = int(input())
# B = input()                  # B를 스트링으로 납둔 상태에서 연산때 int로 변환
# print(A*int(B[2]))
# print(A*int(B[1]))
# print(A*int(B[0]))
# print(A*int(B))