
# --url--
# https://www.acmicpc.net/problem/5086

# --title--
# 5086번: 배수와 약수

# --problem_description--
# 4 × 3 = 12이다.

# 이 식을 통해 다음과 같은 사실을 알 수 있다.

# 3은 12의 약수이고, 12는 3의 배수이다.

# 4도 12의 약수이고, 12는 4의 배수이다.

# 두 수가 주어졌을 때, 다음 3가지 중 어떤 관계인지 구하는 프로그램을 작성하시오.

# --problem_input--
# 입력은 여러 테스트 케이스로 이루어져 있다. 각 테스트 케이스는 10,000이 넘지않는 두 자연수로 이루어져 있다. 마지막 줄에는 0이 2개 주어진다. 두 수가 같은 경우는 없다.

# --problem_output--
# 각 테스트 케이스마다 첫 번째 숫자가 두 번째 숫자의 약수라면 factor를, 배수라면 multiple을, 둘 다 아니라면 neither를 출력한다.

# 1. 두 가지 수를 입력 받는다(두 개 모두 0이 입력 되면 입력 중지) -> while문 사용
# 1) while 에 '0이 아니다'라는 조건을 줘서 거짓이 되면(0을 입력받으면) 입력 중지 
# 2) while문에 조건식을 따로 쓰지 않고 True로 지정하면 무한 루프가 이뤄진다
#    if 문으로 a, b 모두 0인 조건을 주고 break를 걸어주면 루프 중지
#    while문에서 True와 같은 효과 : 0이 아닌 숫자, 내용이 있는 문자열 -< 모두 True로 취급한다

# while문은 조건식이 참일 때 반복, 거짓일 때 반복을 끝낸다
# 특히 while 반복문은 반복 횟수가 정해져 있지 않을 때 자주 사용한다!!!
# 반면 for 반복문은 반복 횟수가 정해져 있을 때 자주 사용한다

# 2. 두 가지 수의 관계를 따져야 함 -> 조건문을 사용 (세 가지의 조건, 약수, 배수, 해당사항없음)
# 약수, 배수 관계이므로 나머지를 구하는 % 연산자 사용

import sys

while True : 
    a, b = map(int, sys.stdin.readline().split())
    if a == 0 and b == 0 :
        break
    elif b % a == 0 :
        print("factor")
    elif a % b == 0 :
        print("multiple")
    else :
        print("neither")
  




# import sys

# a, b = map(int, sys.stdin.readline().split())

# while a != 0 and b != 0 :
#     if b % a == 0 :
#         print("factor")
#     elif a % b == 0 :
#         print("multiple")
#     else :
#         print("neither")
#     a, b = map(int, sys.stdin.readline().split())