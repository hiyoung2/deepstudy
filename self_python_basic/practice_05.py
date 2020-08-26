# 사전, dictionary

cabinet = {3:"김혜수", 100:"김태리"} # 3, 100은 key, "김혜수", "김태리"는 value
print(cabinet[3])
print(cabinet[100])

print(cabinet.get(3))

# print(cabinet[5])
# print("hi")
# 오류 발생으로 프로그램 중지. "hi" 출력되지 않음

# print(cabinet.get(5))
# print("hi")
# line 12 : None 으로 출력, "hi" 출력됨, 프로그램 중지되지 않음

print(cabinet.get(5, "사용가능"))  # 5의 값이 없다면 None 대신 "사용가능"이 출력됨


# 사전자료형 안에 어떤 값이 있는지 확인할 수 있다.

print(3 in cabinet) # 3이라는 key가 cabinet에 있는가? - True
print(5 in cabinet) # 5라는 key가 cabinet에 있는가? - False

cabinet = {"A-3":"강아지", "B-100":"고양이"}
print(cabinet["A-3"])
print(cabinet["B-100"])

# 새 손님이 옴
print(cabinet)
cabinet["C-20"] = "송아지" 
cabinet["A-3"] = "망아지" # A-3에 "강아지"라는 값이 "망아지"라는 값으로 업데이트 됨
print(cabinet)

# 손님이 떠남
del cabinet["A-3"]
print(cabinet)

# key 들만 출력
print(cabinet.keys())

# value 들만 출력
print(cabinet.values())

# key, vaule 쌍으로 출력 , items
print(cabinet.items())

# 목욕탕 폐점
cabinet.clear()
print(cabinet)  # {} 출력
