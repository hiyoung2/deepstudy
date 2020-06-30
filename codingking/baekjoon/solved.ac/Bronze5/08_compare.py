# --title--
# 1330번: 두 수 비교하기

# --problem_description--
# 두 정수 A와 B가 주어졌을 때, A와 B를 비교하는 프로그램을 작성하시오.


# --problem_input--
# 첫째 줄에 A와 B가 주어진다. A와 B는 공백 한 칸으로 구분되어져 있다.


# --problem_output--
# 첫째 줄에 다음 세 가지 중 하나를 출력한다.

# 1. 두 수를 입력 받는다
# 2. if문을 이용하여 조건에 맞게끔 출력한다
# 3. 세 가지 조건이 있으므로 if, elif, else 사용

# a, b = map(int, input().split()) # 공백을 기준으로 입력받으므로 split() 을 쓴다

# if a > b :
#     print('>')
# elif a < b :
#     print('<')
# else :
#     print('==')

# 함수로 만들기 (스터디 참고)
def compare() :
    x, y = map(int, input().split())
    if x > y :
        print(">")
    elif x < y :
        print("<")
    elif x == y :
        print("==")

if __name__ == '__main__' :
    compare()   