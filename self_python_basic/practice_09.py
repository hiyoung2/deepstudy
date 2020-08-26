# for

# ex) 식당에서 아르바이트 중, 손님들이 많이 왔음
# print("대기번호 : 1")
# print("대기번호 : 2")
# print("대기번호 : 3")
# print("대기번호 : 4") 

# 100명이 넘게 왔다면? 
# 100까지 다 입력하기 어려움 

# for로 간편하게 처리할 수 있음
for waiting_no in [0,1,2,3,4]:    # 대기번호가 정해져 있는 경우
    print("대기번호 : {0}".format(waiting_no))

# randrange()
for waiting_no in range(5): # 0,1,2,3,4까지 순차적으로 이루어짐.
     print("대기번호 : {0}".format(waiting_no))

for waiting_no in range(1,6): # 대기번호를 1번부터 쓰고 싶은 경우. 1,2,3,4,5
     print("대기번호 : {0}".format(waiting_no))


# ex) 스타벅스에 손님이 왔다.

starbucks = ["해원", "은섭", "명여"]
for customer in starbucks:
    print("{0}, 커피가 준비되었습니다.".format(customer))