# 리스트, list [] : 순서를 가지는 객체의 집합

# 지하철 칸별로 10명, 20명, 30명이 있다
# subway1 = 10
# subway2 = 20
# subway3 = 30

subway = [10, 20, 30]
print(subway)

subway = ["이소라", "이수영", "헤이즈"]
print(subway)

# 이수영이 몇 번째 칸에 타고 있는가?
print(subway.index("이수영"))  # index는 0부터 시작하므로

# 백예린이 다음 정류장에서 다음 칸에 탐
subway.append("백예린") # append : 더하다, 덧붙이다, 첨가하다 / 가장 마지막에 더해짐
print(subway)

# 거미를 이소라와 이수영 사이에 태워 좀
subway.insert(1, "거미")
print(subway)


# 지하철에 있는 사람을 한 명씩 뒤에서 꺼냄
# print(subway.pop())
# print(subway)

# print(subway.pop())
# print(subway)

# print(subway.pop())
# print(subway)

# 같은 이름의 사람이 몇 명 있는지 확인
subway.append("이소라")
print(subway)
print(subway.count("이소라"))

# 정렬도 가능
num_list = [5,2,4,3,1]
num_list.sort() # 1부터 차례대로 정렬
print(num_list)

# 순서 뒤집기 가능
num_list.reverse() #5부터 반대 순서로 정렬
print(num_list)

# 모두 지우기
num_list.clear()
print(num_list)

# 자료형에 구애 받지 않고 다양하게 섞어서 사용 가능
mix_list = ["이수영", 20, True]
print(mix_list)

# 리스트 확장
num_list = [5,4,3,2,1]

num_list.extend(mix_list)
print(num_list)


