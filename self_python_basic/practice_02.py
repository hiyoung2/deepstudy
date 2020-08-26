# 연산자

print(1+1) # 2
print(3-2) # 1
print(5*2) # 10
print(6/3) # 2

print(2**3) # 2^3 = 8
print(5%3) # 나머지 구하기 2
print(10%3) # 1
print(5//3) # 몫 구하기 1
print(10//3) # 3

print(10 > 3) # True
print(4 >= 7) # False
print(10 < 3) # False
print( 5 <= 5) # True

print(3 == 3) # True
print(4 == 2) # False
print(3 + 4 == 7) # True


print(1 != 3) #True
print(not(1 != 3)) #False

print((3 > 0) and (3 < 5)) # True
print((3 > 0) & (3 < 5)) # True

print((3 > 0) or (3 > 5)) # True
print((3 > 0) | (3 > 5)) # True

print(5 > 4 > 3) # True
print(5 > 4 > 7) # False

# 간단한 수식
print( 2 + 3 * 4) # 14
print( (2 +3) * 4) # 20
number = 2 + 3 * 4
print(number)
number = number + 2 # 16
print(number)
number += 2 # 18
print(number)
number *= 2 # 36
print(number)
number /= 2 # 18
print(number)
number -= 2 # 16
print(number)
number %= 5 # 1
print(number)

# 숫자 처리 함수
print(abs(-5)) # 절댓값
print(pow(4, 2)) # 4^2 , pow(a, n) = a의 n승
print(max(5, 12)) # 최댓값, 12
print(min(5, 12)) # 최솟값, 5
print(round(3.14)) # 반올림, 3
print(round(4.99)) # 반올림, 5

from math import * # math libraray 안의 것을 이용하겠다, 불러오겠다.
print(floor(4.99)) # 내림, 4
print(ceil(3.14)) # 올림, 4
print(sqrt(16)) # 제곱근, 4

from random import * # 랜덤, 무작위로 숫자를 뽑아주는 것 / 난수
# print(random()) # 0.0 이상 1.0 미만의 임의의 값 생성
# print(random() * 10) # 0.0 이상 10.0 미만의 임의의 값 생성
# print(int(random() * 10)) # 0 이상 10 미만의 임의의 값 생성 
# print(int(random() * 10)) # 0 이상 10 미만의 임의의 값 생성 
# print(int(random() * 10)) # 0 이상 10 미만의 임의의 값 생성 
# print(int(random() * 10)) # 0 이상 10 미만의 임의의 값 생성 
# print(int(random() * 10 + 1)) # 1이상 10 이하의 임의의 값 생성
# print(int(random() * 10 + 1)) # 1이상 10 이하의 임의의 값 생성
# print(int(random() * 10 + 1)) # 1이상 10 이하의 임의의 값 생성
# print(int(random() * 10 + 1)) # 1이상 10 이하의 임의의 값 생성
# print(int(random() * 10 + 1)) # 1이상 10 이하의 임의의 값 생성
# print(int(random() * 10 + 1)) # 1이상 10 이하의 임의의 값 생성

# lotto 번호 생성
print(int(random() * 45 + 1)) # 1 ~ 45 이하의 임의의 값 생성
print(int(random() * 45 + 1)) # 1 ~ 45 이하의 임의의 값 생성
print(int(random() * 45 + 1)) # 1 ~ 45 이하의 임의의 값 생성
print(int(random() * 45 + 1)) # 1 ~ 45 이하의 임의의 값 생성
print(int(random() * 45 + 1)) # 1 ~ 45 이하의 임의의 값 생성
print(int(random() * 45 + 1)) # 1 ~ 45 이하의 임의의 값 생성

# print(randrange(1, 45)) # 1 ~ 45 미만의 임의의 값 생성
print(randrange(1, 46)) # 1 ~ 46 미만의 임의의 값 생성
print(randrange(1, 46)) # 1 ~ 46 미만의 임의의 값 생성
print(randrange(1, 46)) # 1 ~ 46 미만의 임의의 값 생성
print(randrange(1, 46)) # 1 ~ 46 미만의 임의의 값 생성
print(randrange(1, 46)) # 1 ~ 46 미만의 임의의 값 생성
print(randrange(1, 46)) # 1 ~ 46 미만의 임의의 값 생성

print(randint(1, 45)) # 1 ~ 45 이하의 임의의 값 생성
print(randint(1, 45)) # 1 ~ 45 이하의 임의의 값 생성
print(randint(1, 45)) # 1 ~ 45 이하의 임의의 값 생성
print(randint(1, 45)) # 1 ~ 45 이하의 임의의 값 생성
print(randint(1, 45)) # 1 ~ 45 이하의 임의의 값 생성
print(randint(1, 45)) # 1 ~ 45 이하의 임의의 값 생성

"""
Quiz) 당신은 최근에 코딩 스터디 모임을 새로 만들었습니다.
월 4회 스터디를 하는데 3번은 온라인으로 하고 1번은 오프라인으로 하기로 했습니다
아래 조건에 맞는 오프라인 모임 날짜를 정해주는 프로그램을 작성하시오.

조건1 : 랜덤으로 날짜를 뽑아야 함
조건2 : 월별 날짜는 다름을 감안하여 최소 일수인 28 이내로 정함
조건3 : 매월 1~3일은 스터디 준비를 해야하므로 제외

(출력문 예제)
오프라인 스터디 모임 날짜는 매월 x 일로 선정되었습니다.
"""

from random import *
# 나의 작성 답안
print("오프라인 스터디 모임 날짜는 매월 " + str(randint(4, 28)) + "일로 선정되었습니다.")


# 강의 답안
date = randint(4, 28)
print("오프라인 스터디 모임 날짜는 매월 " + str(date) + " 일로 선정되었습니다.")

# 변수를 생각하자
