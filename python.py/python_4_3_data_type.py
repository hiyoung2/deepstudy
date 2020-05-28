# 4. 3 자료형
# 4. 3. 1 자료형의 종류

# 파이썬의 값은 형(자료형, data type)이라는 개념이 있다
# 자료형은 문자열형(str형), 정수형(int형), 부동소수점형(float형), 리스트형(list형) 등이 있다

# 다른 형끼리의 게산이나 연결을 시도하면 에러가 발생
# height = 177
# print("신장은 " + height + " cm입니다.")
# TypeError: can only concatenate str (not "int") to str 

height = 177
print(type(height)) # <class 'int'>


h = 1.7
w = 60
print(type(h))
print(type(w))
bmi = w / h**2 # h**2 = h^2
print(bmi)
print(type(bmi))

# 4. 3. 2 자료형의 변환
# 자료에는 다양한 형이 존재한다
# 서로 다른 자료형을 계산하거나 결합하려면 형을 변환하여 같은 형으로 만들어야 한다
# 정수형으로 변환하려면 int(), 소수점을 포함한 수치형으로 변환하려면 float(), 문자열로 변환하려면 str()

h = 177
print("신장은 " + str(h) + "cm입니다.") # 신장은 177cm입니다.

a = 35.4
b = 10
print(a + b)  # 45.4

h = 1.7
w = 60
bmi = w / h**2
print("당신의 bmi는 " + str(bmi) + "입니다.") # 당신의 bmi는 20.761245674740486입니다.

# 4.3.3 자료형의 이해와 확인
# 프로그래밍에서 자료형은 매우 중요하다
# 결국 다른 자료형끼리는 결합할 수 없으며 문자열로 저장된 수치는 계산할 수 없다는 것이다
greeting = "hi!"
print(greeting * 2) # hi!hi! : 문자열이 두 번 나란히 출력

n = "10"
print(n*3)
print(type(n*3)) # <class 'str'>



