# --url--
# https://www.acmicpc.net/problem/16199

# --title--
# 16199번: 나이 계산하기

# --problem_description--
# 한국에서 나이는 총 3가지 종류가 있다.

# 만 나이는 생일을 기준으로 계산한다. 어떤 사람이 태어났을 때, 그 사람의 나이는 0세이고, 생일이 지날 때마다 1세가 증가한다. 
# 예를 들어, 생일이 2003년 3월 5일인 사람은 2004년 3월 4일까지 0세이고, 2004년 3월 5일부터 2005년 3월 4일까지 1세이다.

# 세는 나이는 생년을 기준으로 계산한다. 어떤 사람이 태어났을 때, 그 사람의 나이는 1세이고, 연도가 바뀔 때마다 1세가 증가한다. 
# 예를 들어, 생일이 2003년 3월 5일인 사람은 2003년 12월 31일까지 1세이고, 2004년 1월 1일부터 2004년 12월 31일까지 2세이다.

# 연 나이는 생년을 기준으로 계산하고, 현재 연도에서 생년을 뺀 값이다. 
# 예를 들어, 생일이 2003년 3월 5일인 사람은 2003년 12월 31일까지 0세이고, 2004년 1월 1일부터 2004년 12월 31일까지 1세이다.

# 어떤 사람의 생년월일과 기준 날짜가 주어졌을 때, 기준 날짜를 기준으로 그 사람의 만 나이, 세는 나이, 연 나이를 모두 구하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄에 어떤 사람이 태어난 연도, 월, 일이 주어진다. 생년월일은 공백으로 구분되어져 있고, 항상 올바른 날짜만 주어진다.

# 둘째 줄에 기준 날짜가 주어진다. 기준 날짜도 공백으로 구분되어져 있으며, 올바른 날짜만 주어진다.

# 입력으로 주어지는 생년월일은 기준 날짜와 같거나 그 이전이다.

# 입력으로 주어지는 연도는 1900년보다 크거나 같고, 2100년보다 작거나 같다.

# --problem_output--
# 첫째 줄에 만 나이, 둘째 줄에 세는 나이, 셋째 줄에 연 나이를 출력한다.

# 1. 태어난 연도, 월, 일을 입력 받는다
# 2. if문을 사용해 만 나이, 세는 나이, 연 나이를 계산해서 출력한다
# 3. 세는 나이와 연 나이의 경우에는 그냥 단순 뺄셈이 될 것 같은데 만 나이는 여러 조건을 따져야 할 것 같음
# - 만 나이 
# 출생 연도와 기준이 되는 연도가 같을 경우에는 그냥 0세
# 다를 경우에는 월과 일을 따져서 나이를 구해야 한다고 생각
# - 세는 나이
# 세는 나이의 경우 일단 태어나면 디폴트로 1이기 때문에 기준 연도에서 태어난 연도를 빼고 1을 더하면 될 듯
# - 연 나이
# 연 나이는 태어나면 0세, 연도가 바뀌면 한 살씩 먹으니까 단순 뺄셈?
# 4. 연도, 월, 일을 리스트로 한꺼번에 받아서 리스트 요소별로 빼내서 작업


import sys
birth = list(map(int, sys.stdin.readline().split())) # 태어난 연도, 월, 일을 입력 받음
stdrd = list(map(int, sys.stdin.readline().split())) # 기준 날짜를 입력 받음

# 리스트 추출이 잘 되는지 확인
# print(birth[1]) # 1
# print(type(birth[1])) # <class 'int'>
# tmp = stdrd[0] - birth[0] # 2002, 2008
# print(tmp) # 6

# age_1 : 만 나이
# age_2 : 세는 나이
# age_3 : 연 나이

# 1. 만 나이 구하기
if birth[0] == stdrd[0] :
    age_1 = stdrd[0] - birth[0]
else :
    if birth[1] > stdrd[1] :
        age_1 = stdrd[0] - birth[0] - 1
    elif birth[1] == stdrd[1] and birth[2] > stdrd[2]:
        age_1 = stdrd[0] - birth[0] - 1
    else :
        age_1 = stdrd[0] - birth[0]

print(age_1) # 만 나이


# 2. 세는 나이 구하기
age_2 = stdrd[0] - birth[0] + 1

print(age_2)

# 3. 연 나이 구하기
age_3 = stdrd[0] - birth[0]

print(age_3)