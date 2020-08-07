# --title--
# 10998번: A×B

# --problem_description--
# 두 정수 A와 B를 입력받은 다음, A×B를 출력하는 프로그램을 작성하시오.


# --problem_input--
# 첫째 줄에 A와 B가 주어진다. (0 < A, B < 10)


# --problem_output--
# 첫째 줄에 A×B를 출력한다.


# 1. 두 개의 수를 입력 받는다
# 2. 두 개의 숫자를 곱한 결과를 출력

# a, b = map(int, input().split()) # 공백을 기준으로 입력받으므로 split() 을 쓴다

# print(a*b) # 두 개의 수를 곱한 것을 출력

# 함수로 만들기 (스터디 참고)
def multiply() :
    x, y = map(int, input().split())
    z = x * y
    print(z)

# if __name__ == '__main__' :
#     multiply()

multiply()