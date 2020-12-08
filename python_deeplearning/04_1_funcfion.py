# 함수 - function

# 함수 정의
# def 함수이름(입력1, 입력2, ...): -> 
# 입력1 == 입력파라미터
# 파이썬에서는 입력 파라미터의 데이터 타입을 따로 명시하지 않는다

# 함수 반환값
# 파이썬 함수는 한 개 이상의 return 값을 반환 할 수 있다
# return 값은 콤마(,)로 분리하여 받거나 tuple 형태로 받을 수 있다

def multi_ret_func(x):
    return x+1, x+2, x+3 # return (x+1, x+2, x+3)

x = 100
y1, y2, y3 = multi_ret_func(x)

print(y1, y2, y3)
print()
# default parameter
# 함수의 입력 파라미터에 기본 값을 지정하는 것을 말함
# 이렇나 디폴트 파라미터는 함수가 호출되었을 경우 입력 파라미터에 명시적인 값이 전달되지 않으면
# 기본으로 지정한 값을 사용하겠다는 의미

# example
def print_name(name, count=2):

    for i in range(count):
        print("name ==", name)

print_name("YOUNG")
print()
print_name("YOUNG", 5)
print()
# mutable / immutable parameter
# 입력 파라미터가 mutable(list, dict, numpy etc.) 데이터형인 경우는 원래의 데이터에 변형이 일어남
# immutable(숫자, 문자, tuple etc.)은 원래의 데이터에 변형이 일어나지 않음

# int_x : immutable / input_list : mutable
def mutable_immutable_func(int_x, input_list):

    int_x += 1
    input_list.append(100)

x = 1 # immutable
test_list = [1, 2, 3] # mutable

mutable_immutable_func(x, test_list)

print("x ==", x, ", test_llist ==", test_list)
# int는 변하지 않고 x는 1이 나옴
# list는 함수를 거쳐서 돌아오면 100이 append 된 상태이다