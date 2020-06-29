# 5522 : 카드 게임

# 입력
# 표준 입력에서 다음과 같은 데이터를 읽어온다.
# i 번째 줄(1 ≤ i ≤ 5)에는 정수 Ai가 적혀있다. 이것은 i번째 게임에서의 JOI군의 점수를 나타낸다.
# 모든 입력 데이터는 다음 조건을 만족한다.
# 0 ≤ Ai ≤ 100．

# 출력
# 표준 출력에 JOI군의 총점을 한 줄로 출력하라.

# 총 5번의 게임 점수를 입력 받아야 한다
# 따라서 input을 이용하여 5번의 입력을 받는데
# 정수형으로 받아야 하므로 int로 감싸준다

# a = int(input())
# b = int(input())
# c = int(input())
# d = int(input())
# e = int(input())

# 출력은 총점이므로 입력받은 각 점수들의 합을 sum 이라는 변수에 대입
# sum = a + b + c + d + e

# 총점 출력
# print(sum)

# for문으로 한 번에 처리하기

# cnt = range(5)
total = 0

for i in range(5) :
    score = int(input())
    total += score
print(total)