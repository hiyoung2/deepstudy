
# --url--
# https://www.acmicpc.net/problem/2754

# --title--
# 2754번: 학점계산

# --problem_description--

# 	어떤 사람의 C언어 성적이 주어졌을 때, 평점은 몇 점인지 출력하는 프로그램을 작성하시오.

# 	A+: 4.3, A0: 4.0, A-: 3.7
# 	B+: 3.3, B0: 3.0, B-: 2.7
# 	C+: 2.3, C0: 2.0, C-: 1.7
# 	D+: 1.3, D0: 1.0, D-: 0.7
# 	F: 0.0
# --problem_input--
# 	첫째 줄에 C언어 성적이 주어진다. 성적은 문제에서 설명한 13가지 중 하나이다.
# --problem_output--
# 	첫째 줄에 C언어 평점을 출력한다.

# 1. 성적 입력 받기 -> A0 이런 식으로 입력받기 때문에 문자열 (input() default : str)
# 2. if 조건문을 사용해서 각 점수에 맞게끔 평점 출력

# 성적 기준 : 각 4, 3, 2, 1 점에서 +3이면 +, -3이면 -
# A0 == 4.0, B0 == 3.0, C0 == 2.0, D0 == 1.0을 기준으로 잡으면 덜 복잡하게 조건문 가능?
# 문자열에서 A만 가져와서 4로 지정, B는 3으로... -> 기준점 설정
# 문자열 슬라이싱 -> 문자열을 이루는 각 문자들의 index를 이용


# 1차 시도 - 런타임 에러
grade = input()

print(grade) # A0
print(type(grade)) # <class 'str'>
print(grade[0]) # A

if grade[0] == "A" :
    score = 4.0
elif grade[0] == "B" :
    score = 3.0
elif grade[0] == "C" :
    score = 2.0
elif grade[0] == "D" :
    score = 1.0
else :
    score = 0.0

# print(score)

if grade[1] == "+" : # grade[1] == + or - (0은 제외, 위의 조건문이 있으므로)
    score += 0.3
elif grade[1] == "-" :
    score -= 0.3

print(score)