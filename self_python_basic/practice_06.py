# 튜플(tuple)
# 내용의 변경이나 추가 불가능, 리스트보다 속도가 빠름

# 딱 2가지 메뉴만 파는 돈까스집 하나가 있다.
menu = ("돈까스", "치즈까스")
print(menu[0])
print(menu[1])

# menu.add("생선까스") : 오류발생 / 추가가 불가능.

name = "아무개"
age = 20
hobby = "코딩"
print(name, age, hobby)

(name, age, hobby) = ("아무개", 20, "코딩")
print(name, age, hobby)


# 세트 (set) , 집합
# 중복 안 됨, 순서 없음
my_set = {1,2,3,3,3}
print(my_set) # {1,2,3} 으로 출력, 중복이 안 되므로

java = {"제이크", "에이미", "지나"}
python = set(["제이크", "로사"])

# 교집합 (java와 python을 모두 할 수 있는 개발자)
print(java & python)
print(java.intersection(python))

# 합집합 (java 또는 python을 할 수 있는 개발자)
print(java | python)
print(java.union(python)) # 순서는 상관없이 출력

# 차집합 (java 가능, python 불가능한 개발자)
print(java - python)
print(java.difference(python))

# python 할 줄 아는 사람이 늘어남
python.add("에이미")
print(python)

# java를 까먹은 사람이 생김
java.remove("에이미")
print(java)