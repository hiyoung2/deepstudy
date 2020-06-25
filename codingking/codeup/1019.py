# 연원일 입력받아 그대로 출력하기

a, b, c = map(int, input().split("."))
print("%004d.%02d.%02d" % (a, b, c))