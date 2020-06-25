# 5554 : 심부름 가는 길

# 입력
# 입력은 총 4줄이며, 한 줄에 하나씩 양의 정수가 적혀있다.
# 첫 번째 줄에 집에서 학교까지의 이동 시간을 나타내는 초가 주어진다.
# 두 번째 줄에 학교에서 PC방까지의 이동 시간을 나타내는 초가 주어진다.
# 세 번째 줄에 PC방에서 학원까지의 이동 시간을 나타내는 초가 주어진다. 
# 마지막 줄에 학원에서 집까지의 이동 시간을 나타내는 초가 주어진다.
# 집에 늦게 가면 혼나기 때문에, 총 이동시간은 항상 1 분 0 초 이상 59 분 59 초 이하이다.

# 출력
# 총 이동시간 x 분 y 초를 출력한다. 첫 번째 줄에 x를, 두 번째 줄에 y를 출력한다.

# 각 4개의 시간을 입력받아야 하므로 input을 4번 사용해 입력 받음
to_school = input()
to_pcroom = input()
to_class = input()
to_home = input()

# 연산하기 위해 정수형으로 변환
to_school = int(to_school)
to_pcroom = int(to_pcroom)
to_class = int(to_class)
to_home = int(to_home)

# 분 / 초 단위로 나눠서 출력해야하므로 초의 합을 hour 변수에 넣고
# 최종 '분'출력을 위해 필요한 60을 sec 이라는 변수에 대입하여 미리 준비
hour = to_school + to_pcroom + to_class + to_home
sec = 60

print(hour//sec) # '분(minute)' 출력
print(hour%sec)  # '초(seconds)' 출력

# 이것도 너무 복잡한 느낌(변수명을 너무 길게 쓴 탓인가)
