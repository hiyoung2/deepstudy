# 숫자와 단순연산
print(5)
print(-10)
print(3.14)
print(1000)
print(5+3)
print(2*8)
print(3*(3+1))

# 문자열 자료형
print('풍선')
print("나비")
print("ㅋㅋㅋㅋㅋㅋㅋㅋㅋ")
print("ㅋ"*9)

# boolean 자료형
# 참과 거짓
print(5 > 10)
print(5 < 10)
print(True)
print(False)
print(not True) # True의 반대
print(not False)
print(not ( 5 > 10))

# 변수
# 요청 :  반려 동물을 소개해 주세요

animal = "강아지"
name = "구름이"
age = 4 #정수형 자료이므로 따옴표 없이 숫자를 바로 적음
hobby =  "산책"
is_adult = age >= 3     # age가 3 이상이면 True 아니면 False


print("우리집 " + animal + "의 이름은 " + name + "예요")
hobby = "간식"
# print(name + "는 " + str(age) +"살이며, " + hobby + "을 아주 좋아해요")
print(name, "는 ", age , "살이며, ", hobby, "을 아주 좋아해요") 
#','로 '+'처럼 연결 가능하지만 옆 문자열과 공백이 발생함
print(name + "는 어른일까요? " + str(is_adult))

# 주석
'''이렇게
하면
여러 문장이
주석처리
됩니다
'''
# 여러 문장
# 주석처리
# 또 다른 방법
# 주석처리 할 문장을 모두 선택 후 ctrl + /
# 주석처리 된 문장을 모두 선택 후 ctrl + / 하면 주석 해제됨

# Quiz) 변수를 이용하여 다음 문장을 출력하시오.
# 변수명 : station
# 변수값 : "사당", "신도림", "인천공항" 순서대로 입력
# 출력 문장 : XX행 열차가 들어오고 있습니다.

station = "사당"
print(station + "행 열차가 들어오고 있습니다.")

station = "신도림"
print(station + "행 열차가 들어오고 있습니다.")

station = "인천공항"
print(station + "행 열차가 들어오고 있습니다.")




















