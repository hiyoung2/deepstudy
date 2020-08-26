# while
# ex) 커피가 준비된 상태에서 손님 이름 확인, 다른 손님이 찾아오는 경우 준비됨을 계속 알림.

customer = "푸우"    # 커피가 준비된 사람
person = "Unknown"   # 종업원에게 찾아온 사람

while person != customer :  # 커피가 준비된 사람이 아닐 경우 계속 반복
    print("{0}, 커피가 준비 되었습니다.".format(customer))
    person = input("이름이 어떻게 되세요?")     # 손님의 이름 입력

    # '푸우'가 입력 되면 while문 탈출, 프로그램 종료.