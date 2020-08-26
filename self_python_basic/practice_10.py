# while

# ex) 스타벅스에서 손님을 5번 불러도 오지 않으면 커피를 버리는 정책 시행
customer = "미피"
index = 5
while index >=1:    # 조건을 만족할 때까지 반복, while 옆에 조건을 입력.
    print("{0}, 커피가 준비 되었습니다. {1} 번 남았어요.".format(customer, index))
    index -= 1    # 한 번씩 줄여나감
    if index == 0:
        print("커피는 폐기처분되었습니다.")