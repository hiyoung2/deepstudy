# 자료구조의 변경
menu = {"커피", "우유", "주스"}
print(menu)
print(menu, type(menu)) # class : set / {}

menu = list(menu)
print(menu, type(menu)) # class : list / []

menu = tuple(menu)
print(menu, type(menu)) # class : tuple / ()

menu = set(menu)
print(menu, type(menu))


"""
Quiz) 
당신의 학교에서는 파이썬 코딩 대회를 주최합니다.
참석률을 높이기 위해 댓글 이벤트를 진행하기로 하였습니다.
댓글 작성자들 중에 추첨을 통해 1명은 치킨, 3명은 커피 쿠폰을 받게 됩니다.
추첨 프로그램을 작성하시오.

조건1 : 편의상 댓글을 20명이 작성하였고 아이디는 1~20 이라고 가정
조건2 : 댓글 내용과 상관 없이 무작위로 추첨하되 중복 불가
조건3 : random 모듈의 suffle 과 sample을 활용

(출력 예제)
 -- 당첨자 발표 --
 치킨 당첨자 : 1
 커피 당첨자 : [2, 3, 4]
 -- 축하합니다 --

(활용 예제)
from random import *
lst = [1,2,3,4,5]
print(lst)
suffle(lst)
print(lst)
pritn(sample(lst,1))
"""

# lst = [1,2,3,4,5]
# print(lst)
# shuffle(lst) : lst에 있는 값들을 무작위로 위치를 바꿈
# print(lst)
# print(sample(lst, 1)) : lst에서 한 개를 무작위로 뽑음




# 나의 작성 답안
from random import *

lst = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#print(lst)
shuffle(lst)
#print(lst)

print("-- 당첨자 발표 --")
print("치킨 당첨자 : " +str(sample(lst, 1)))
print("커피 당첨자 : " +str(sample(lst, 3)))
print("-- 축하합니다 --")

# lst = range(1, 21) 써 봤지만 오류 계속 발생, 1부터 20까지 직접 타이핑
# suffle 쓰기 위해서 list 타입으로 바꿔줘야 함을 몰랐음
# lst의 원래 타입은 range, list 타입으로 변경해줘야 함
# line 57 : 한 명 뽑고, 3명 뽑으면 중복 가능성이 있음





# 강의 답안
# from random import *
users = range(1, 21)
# print(type(users))
users = list(users)
# print(type(users))

print(users)
shuffle(users)
print(users)

winners = sample(users, 4) # 중복 방지를 위해 일단 4명을 뽑음, 4명 중 1명은 치킨, 3명은 커피

print(" -- 당첨자 발표 -- ")
print("치킨 당첨자 : {0}".format(winners[0])) # winners 중 index 0번째
print("커피 당첨자 : {0}".format(winners[1:])) # winners 중 index 1번째부터 끝까지, 총 3명
print(" -- 축하합니다 -- ")











