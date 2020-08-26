# 한 줄 for 활용

# 출석번호가 1, 2, 3, 4,5 였는데 앞에 100을 붙이는 걸로 규칙을 바꿈. 
# 즉, 101, 102, 103, 104, 105로 바뀜.

students = [1,2,3,4,5]
print(students)
students = [i+100 for i in students]
print(students)

# 영화 속 캐릭터 이름을 길이로 변환
movie = ["Elizabeth", "Darcy", "Jane"]
movie = [len(i) for i in movie]
print(movie)

# 드라마(Brooklyn Nine-nine) 속 캐릭터 이름을 대문자로 변환
bnana = ["Amy", "Gina", "Rosa"]
bnana = {i.upper() for i in bnana}
print(bnana)