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

import sys

color1 = sys.stdin.readline()
color2 = sys.stdin.readline()
color3 = sys.stdin.readline()

# print(color1 + color2 + color3[1:])

