# 함수 - lambda
# lambda는 한 줄로 함수를 작성하는 방법
# anonymouse(익명) 함수 또는 lambda expression 람다 표현식 등으로 불림

# lambda는 다른 함수의 인수(parameter)로 넣을 때 주로 사용
# 특히 머신러닝에서 미분을 계산하기 위해 필요한 수치 미분과 활성화 함수 등을 표현할 때
# 수학에서 함수 표현처럼 f(x), f(x,y) 등으로 사용된다

# 함수명 = lambda 입력1, 입력 2, ... : 대체되는 표현식

# lambda 함수 선언
f = lambda x : x + 100 # 수학 함수표현 f(x)를 x+100 으로 대체된다는 의미

for i in range(3):
    print(f(i)) 


###################################################################


# lambda 에서 입력값을 반드시 이용할 필요는 없다

def print_hello():
    print("hello python")

def test_lambda(s, t):
    print("input1 ==", s, ", input2 ==", t)

s = 100
t = 200
# 생각해보기 -> s, t 선언 및 값 할당이 없으면 error
fx = lambda x, y : test_lambda(s, t) # fx(x, y) = test_lambda(s, t)
fy = lambda x, y : print_hello() # fy(x, y) = print_hello()
# -> fx == 입력으로 x, y를 받아 최종적으로 test_lambda(s, t) 함수로 대체된다
# -> fy == 입력으로 x, y를 받아 최종적으로 print_hello() 함수로 대체된다

fx(500, 1000)
fy(300, 600)

# input1 == 100 , input2 == 200
# hello python

# 수학적 표현을 하기 위해 일반적으로 쓰는 것이 lambda 함수이다
