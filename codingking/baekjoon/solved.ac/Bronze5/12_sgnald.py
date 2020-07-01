# --title--
# 5543번: 상근날드

# --problem_description--
# 상근날드에서 가장 잘 팔리는 메뉴는 세트 메뉴이다. 주문할 때, 자신이 원하는 햄버거와 음료를 하나씩 골라, 세트로 구매하면, 
# 가격의 합계에서 50원을 뺀 가격이 세트 메뉴의 가격이 된다.
# 햄버거와 음료의 가격이 주어졌을 때, 가장 싼 세트 메뉴의 가격을 출력하는 프로그램을 작성하시오.

# --problem_input--
# 입력은 총 다섯 줄이다. 첫째 줄에는 상덕버거, 둘째 줄에는 중덕버거, 셋째 줄에는 하덕버거의 가격이 주어진다. 
# 넷째 줄에는 콜라의 가격, 다섯째 줄에는 사이다의 가격이 주어진다. 모든 가격은 100원 이상, 2000원 이하이다.

# --problem_output--
# 첫째 줄에 가장 싼 세트 메뉴의 가격을 출력한다.


# sdburger = int(input())
# jdburger = int(input())
# hdburger = int(input())
# cocacola = int(input())
# cheonyeon = int(input())
# 버거는 버거끼리, 음료는 음료끼리 따로 입력을 받고 하나의 묶음으로 만들어줘야 할 것 같아서 수정

burgers = []
for i in range(3) :
    burger = int(input())
    burgers.append(burger)
# print(burgers)

drinks = []
for j in range(2) :
    drink = int(input())
    drinks.append(drink)
# print(drinks)

# "파이썬 가장 작은 수 찾기" 구글링으로 
# 파이썬에서 제공하는 min 함수로 리스트(또는 튜플)에서 가장 작은 값을 구할 수 있는 방법을 배움
# cheapest_burger = min(burgers) # 반대로 가장 큰 값을 구하려면 max()함수를 사용하면 된다
# cheapest_drink = min(drinks)

set_menu = min(burgers) + min(drinks) - 50
print(set_menu)