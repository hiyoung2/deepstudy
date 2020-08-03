# 9498 : 시험 성적
# --url--
# https://www.acmicpc.net/problem/9498

# --title--
# 9498번: 시험 성적

# --problem_description--
# 시험 점수를 입력받아 90 ~ 100점은 A, 80 ~ 89점은 B, 70 ~ 79점은 C, 60 ~ 69점은 D, 나머지 점수는 F를 출력하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄에 시험 점수가 주어진다. 시험 점수는 0보다 크거나 같고, 100보다 작거나 같은 정수이다.

# --problem_output--
# 시험 성적을 출력한다.

# score = input()
# score = int(score)

score = int(input())

if score>=90 and score<=100 :
    print("A")
elif score>=80 and score<90 :
    print("B")
elif score>=70 and score<80 :
    print("C")
elif score>=60 and score<70 :
    print("D")
else :
    print("F")
    