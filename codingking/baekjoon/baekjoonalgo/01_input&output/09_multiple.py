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