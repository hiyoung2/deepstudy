# 문자열, string

sentence = '나는 소년입니다.'
print(sentence)
sentence2 = "파이썬은 쉬워요."
print(sentence2)
sentence3 = """
나는 소년이고,
파이썬은 쉬워요.
"""
print(sentence3)

# 슬라이싱 (Slicing) : 원하는 정보만 추출 할 수 있음

jumin = "930717-1234567" # 항상 0부터 시작 / index

print("성별 : " + jumin[7]) # 7번째 자리
print("연 : " + jumin[0:2]) # 0번째부터 2번째 아랫자리까지 가져옴 (0, 1)
print("월 : " + jumin[2:4]) # 2번째부터 4번째 아랫자리까지 가져옴 (2, 3)
print("일 : " + jumin[4:6]) # 4번째부터 6번째 아랫자리까지 가져옴 (4, 5)
print("생년월일 : " + jumin[:6]) # 처음부터 6번째 아랫자리까지 가져옴 (0, 1, 2, 3, 4, 5)
print("뒤 7자리 : " + jumin[7:]) # 7번째부터 끝까지 가져옴 (7, 8, 9, 10, 11, 12, 13)
print("뒤 7자리 (뒤에부터) : " + jumin[-7:])
# 뒷자리 7 : -1, 6 : -2, 5 : -3 ... 1 : -7

# 문자열 처리함수
python = "Python is Amazing"
print(python.lower()) # 문자열 소문자로 출력
print(python.upper()) # 문자열 대문자로 출력
print(python[0].isupper()) # 0번째 문자가 대문자인가? 대문자이면 True, 아니면 False
print(len(python)) # 문자열의 길이 (공백 포함)
print(python.replace("Python", "Java")) # Python을 Java로 대체

index = python.index("n")
print(index) # 가장 처음 나타나는 문자의 자리만 표현

index = python.index("n", index + 1) # 첫번째 찾은 n의 위치에 1을 더한 6번째자리부터 n을 찾음
print(index) # 문자열 중 두 번째로 나타나는 n의 자리

print(python.find("n"))
print(python.find("Java")) # find 에서는 해당 값(Java)이 없을 경우 -1 값을 반환
#print(python.index("Java")) # index 에서는 해당 값이(Java) 없을 경우 오류 발생하면서 프로그램을 종료시킴
print("hi")
"""
find와 index의 차이
42번의 오류 때문에 hi가 출력이 되지 않고 그 전에 프로그램 종료 됨
42번을 주석처리하면 hi는 출력 됨. find는 프로그램을 종료시키지 않기 때문.
"""
print(python.count("n")) # n이 총 몇 번 등장하는가?

# 문자열 포맷

print("a" + "b")
print("a", "b")

# 방법 1
print("나는 %d살입니다." % 20) # %d : 정수
print("나는 %s을 좋아해요." % "파이썬") # %s : 문자열, string값
print("Apple은 %c로 시작해요." % "A") # %c(character), 하나의 문자만 받음

print("나는 %s살입니다" % 20) # %s는 문자뿐만 아니라 다른 것도 받을 수 있음
print("나는 %s색과 %s색을 좋아해요." % ("보라", "검정"))

# 방법 2
print("나는 {}살입니다.".format(20))
print("나는 {}색과 {}색을 좋아해요".format("파랑", "아이보리"))
print("나는 {0}색과 {1}색을 좋아해요".format("노랑", "분홍")) # '노랑'을 0번째, '분홍'을 1번째로 인식
print("나는 {1}색과 {0}색을 좋아해요".format("노랑", "분홍")) # '노랑'을 0번째, '분홍'을 1번째로 인식, 순서 바뀌어서 출력

# 방법 3
print("나는 {age}살이며, {color}색을 좋아해요.".format(age = "20", color = "빨강")) 
print("나는 {age}살이며, {color}색을 좋아해요.".format(color = "빨강", age = "20")) # 변수 이름에 맞게 들어감


# 방법4
age = 20
color = "하늘"
print(f"나는 {age}살이며, {color}색을 좋아해요.")


# 탈출문자

# \n : 줄바꿈
print("백문이 불여일견 \n백견이 불여일타") 

# \" : ", \' : '
# 저는 "코딩입문자"입니다.
print("저는 \"코딩입문자\"입니다")
print("저는 \'코딩입문자\'입니다")

# \\ : \
print("C:\\Users\\inyoung\\Desktop\\PythonWorkspace>")

# \r : 커서를 맨 앞으로 이동
print("Red Apple\rPine") # 커서가 맨 앞으로 이동해서 "Red "까지를 Pine으로 대체


# \b : 백스페이스, Backspace (한 글자 삭제)
print("Redd\bApple") # d 하나를 지움

# \t : 탭, Tab (4칸, 8칸... 공백)

print("Red\tApple")


"""
Quiz) 사이트별로 비밀번호를 만들어주는 프로그램을 작성하시오

예) http://naver.com
규칙1 : http:// 부분은 제외 => naver.com
규칙2 : 처음 만나는 점(.) 이후 부분은 제외 => naver
규칙3 : 남은 글자 중 처음 세자리 + 글자 갯수 + 글자 내 'e' 갯수 + "!"로 구성
                    (nav)           (5)           (1)         (!)
예) 생성된 비밀번호 : nav51!
"""

# 나의 작성답안
site = "http://google.com"
password = ((site[7:10])) + str((len(site[7:-4]))) + str((site[7:-4].count("e"))) + "!"
print("{0} 의 비밀번호는 {1}입니다.".format(site, password))

# 강의 답안
url = "http://naver.com"
my_str = url.replace("http://","") # 규칙1, http:// 는 공백으로 replace
#print(my_str) # naver.com만 출력됨
my_str = my_str[:my_str.index(".")] # 규칙2, my_str[0:5]를 의미, 0부터 5번째 직전 자리까지(0,1,2,3)
#print(my_str) #naver만 출력됨, '.'직전까지

password = my_str[:3] + str(len(my_str) ) + str(my_str.count("e")) + "!"
print("{0} 의 비밀번호는 {1} 입니다.".format(url, password))
