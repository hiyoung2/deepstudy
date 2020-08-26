# continue & break

"""
ex) 출석부 번호대로 학생들에게 책을 읽으라고 시킴
    2번, 5번은 결석인 상태, 결석 학생들은 차례를 넘어가야 함
"""

absent = [2, 5]
no_book = [7] # 책을 깜빡 했음.
for student in range(1, 11):     # 1번부터 10번까지 총 10명의 학생이 있음.
    if student in absent: 
        continue        # 2번, 5번일 때는 print를 실행하지 않고 다음 번호로 넘어감.
    elif student in no_book:
        print("오늘 수업 여기까지. {0}번은 교무실로 따라와".format(student))
        break           # 뒤에 무엇이 있든 바로 반복문 탈출.
    print("{0}번, 책 읽어보세요.".format(student))
