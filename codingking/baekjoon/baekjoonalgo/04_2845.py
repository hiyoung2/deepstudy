

num, area = map(int, input().split())
a, b, c, d, e = map(int, input().split())

# 1차 시도 : 실패
# print(((num*area)-a), ((num*area)-b), ((num*area)-c), ((num*area)-d), ((num*area)-e))


# 2차 시도 : 실패
# tmp = [a, b, c, d, e]
# print(tmp)
# for i in tmp :
#     result = num*area - i
#     print(result)
  
# 3차 시도 : 실패, 역시 안 될 줄 알았음
# pred = num*area

# a_1 = pred-a
# b_1 = pred-b
# c_1 = pred-c
# d_1 = pred-d
# e_1 = pred-e

# print(a_1, b_1, c_1, d_1, e_1)

# 뭔가 더 간단하게 만들어야 하는 것 같은데
