import sys
n, m = map(int, sys.stdin.readline().split())
cnt = n * m
# result = []

for i in range(cnt) :
    i += 1
    # result.append(i)
    print(i)
    # if i % m == 0 :
    # print(i, end = ' ')
    # if i % m == 0 :


# 나머지가 0일 때를 생각해서 이분법으로 딱 되게 

# for j in range(cnt) :
#     if result[j] % m == 1 :
#         print()
#     print(result[j], end = ' ')

# for j in range(cnt) :
#     if result[j] % m == 1 :
#         print('\n')
#     print(result[j], end = ' ')

# for j in range(cnt) :
#     if result[j] % m == 0 :
#         print(result[j], end = ' ')