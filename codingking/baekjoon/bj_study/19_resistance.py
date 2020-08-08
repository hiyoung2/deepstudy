# --url--
# https://www.acmicpc.net/problem/1076

# --title--
# 1076번: 저항

# --problem_description--
# 전자 제품에는 저항이 들어간다. 저항은 색 3개를 이용해서 그 저항이 몇 옴인지 나타낸다.

# 처음 색 2개는 저항의 값이고, 마지막 색은 곱해야 하는 값이다.

# 저항의 값은 다음 표를 이용해서 구한다.

# 예를 들어, 저항에 색이 yellow, violet, red였다면 저항의 값은 4,700이 된다.

# --problem_input--
# 첫째 줄에 첫 번째 색, 둘째 줄에 두 번째 색, 셋째 줄에 세 번째 색이 주어진다. 색은 모두 위의 표에 쓰여 있는 색만 주어진다.

# --problem_output--
# 입력으로 주어진 저항의 저항값을 계산하여 첫째 줄에 출력한다.

# import sys

# color1 = sys.stdin.readline()
# color2 = sys.stdin.readline()
# color3 = sys.stdin.readline()

# print(color1 + color2 + color3[1:])

# color의 값?들을 먼저 설정해줘야 한다
# 표를 보면 color마다 값들이 있는데 이를 index로 사용할 수 있다
# black의 index == 0, brown's index == 1,...
# 색깔들을 리스트에 담아본다

colors = ["black", "brown", "red", "orange", "yellow", "green",
            "blue", "violet", "grey", "white"]

# 리스트 인덱스 값을 이용해서 저항값을 도출 할 수 있다
# 색깔을 입력하면 인덱스가 저장되는 변수들을 만든다

color1 = colors.index(input()) 
# 어떤 요소의 리스트 안에서 위치를 알고 싶을 때 index()를 사용
# ex. black의 위치를 알고 싶으면
# black_index = colors.index("black") 이라 쓴다
# 문제를 풀기 위해 "black"을 미리 명시하지 않고
# 입력 받아야 하므로 input()을 쓴다
# print(color1ye) # yello를 입력했을 시에 4가 출력됨을 확인

color2 = colors.index(input())
color3 = colors.index(input())

# 값이 0, 곱은 1, 1-10, 2-100
# 값과 곱이 10의 지수와 관계있음!
# 따라서 문제의 정답을 구하려면

# print(type(color1)) # int
# color1과 color2 값을 이어 붙인 뒤에
# color3을 10의 지수로 이용하여
# 두 개를 곱하여주면 문제의 답을 도출할 수 있다

result = int(str(color1) + str(color2)) * (10 ** color3)
# color1과 color2의 값을 각각 str으로 변형 후
# 단순하게 병합 시켜주고
# color3과의 계산을 위해 다시 int형으로 변환시켜준다

print(result)

