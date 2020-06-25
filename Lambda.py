# lambda, 람다 
# 파이썬에서 런타임에 생성해서 사용할 수 있는 익명 함수
# filter(), map(), reduce() 와 같은 전형적인 기능 개념(?)과 함께 사용되는 매우 강력한 개념
# lambda는 쓰고 버리는 일시적인 함수
# 함수가 생성된 곳에서만 필요하다
# 즉, 간단한 기능을 일반적인 함수와 같이 정의해 두고 쓰는 것이 아니라, 필요한 곳에서 즉시 사용하고 버릴 수 있다

# 함수는 원래 def로 정의해서 사용했으나 람다를 통해 익명함수를 만들 수 있다
# 람다 표현식은 '식' 형태로 되어 있다고 해서 lambda expression, 람다 표현식이라고도 부른다
# 특히 람다 표현식은 함수를 간편하게 작성할 수 있어서 다른 함수의 인수로 넣을 때 주로 사용한다!!!


#############################################################################################################
# 람다 표현식으로 함수 만들기
# 먼저 람다 표현식을 사용하기 전에 숫자를 받은 뒤에 10을 더해서 반환하는 함수 plus_ten 만들어보기

def plus_ten(x) :
    return x + 10

print(plus_ten(1)) # 11

# 위의 plus_ten 함수를 람다 표현식으로 작성해보자
# 람다 표현식은 다음과 같이 lambda에 매개변수를 지정하고 :(콜론) 뒤에 반환값으로 사용할 식을 지정한다
# -> lambda 매개변수들: 식 (인자리스트: 표현식 / 인자: 표현식)

print(lambda x: x+10)
# 이를 실행하면
# <function <lambda> at 0x000001E212181EE8>
# 함수 객체가 나오는데, 위와 같은 상태로는 함수를 호출할 수가 없기 때문이다
# 람다 표현식은 이름이 없는 함수를 만들기 때문! -> 익명 함수(anonymous function)로 부르는 이유

# lambda로 만든 익명 함수를 호출하려면 다음과 같이 람다 표현식을 변수에 할당해주면 된다
add_ten = lambda x: x+10
print(add_ten(1)) # 11

# 위의 람다 표현식을 살펴보면 lambda x: x + 10 은 매개변수 x 하나를 받고, x에 10을 더해서 반환하다는 뜻
# 즉, 매개변수, 연산자, 값 등을 조합한 식으로 반환값을 만드는 방식이다


#############################################################################################################
# 람다 표현식 자체를 호출하기
# 람다 표현식은 변수에 할당하지 않고 람다 표현식 자체를 바로 호출할 수 있다 (wow, 변수 설정을 안 해줘도 된다?)
# 다음과 같이 람다 표현식을 () 괄호로 묶은 뒤에 다시 ()를 붙이고 인수를 넣어서 호출하면 된다

print((lambda x: x+10)(1)) # 11


#############################################################################################################
# 람다 표현식 안에서는 변수를 만들 수 없다
# 람다 표현식에서 주의할 점은 람다 표현식 안에서는 새 변수를 만들 수 없다는 점
# 따라서 반환값 부분은 변수 없이 식 한줄로 표현할 수 있어야 한다
# 변수가 필요한 코드일 경우에는 def로 함수를 작성하는 것이 좋다

# print(lambda x: y = 10; x+y)(1) # SyntaxError: invalid syntax(문법 에러)

# 단, 아래와 같이 람다 표현식 바깥에 있는 변수는 사용할 수 있다
y = 10
print((lambda x: x+y)(1)) # 11


#############################################################################################################
# 람다 표현식을 인수로 사용하기
# 람다 표현식을 사용하는 이유는 함수의 인수 부분에서 함수를 만들기 위해서이다
# 이런 방식으로 사용하는 대표적인 예가 'map'이다

# 람다 표현식을 사용하기 전에 먼저 def로 함수를 만들어서 map을 사용해보자
# 다음과 같이 숫자를 받은 뒤 10을 더해서 반환하는 함수 plus_ten을 작성
# 그리고 map에 plus_ten 함수와 리스트 [1, 2, 3]을 넣는다
# 물론 map의 결과는 map 객체이므로 눈으로 확인할 수 있도록 list를 사용하여 리스트로 변환해준다

def p_ten(x) :
    return x + 10
print(list(map(p_ten, [1, 2, 3]))) # [11, 12, 13]

# 지금까지 map을 사용할 때 map(str, [1, 2, 3])과 같이 자료형 int, float, str 등을 넣었겠지만
# 사실 p_ten처럼 함수를 직접 만들어서 넣어도 된다

# 람다 표현식으로 함수를 만들어서 map에 활용
print(list(map(lambda x:x+10, [1, 2, 3]))) # [11, 12, 13]

# p_ten 대신 람다 표현식을 사용하니까 코드가 3줄에서 1줄로 줄었다 (!!)
# 람다 표현식은 함수를 다른 함수의 인수로 넣을 때 매우 편리하다


#############################################################################################################
# 람다 표현식으로 매개변수가 없는 함수 만들기
# 람다 표현식으로 매개변수가 없는 함수를 만들 때는 lambda 뒤에 아무것도 지정하지 않고
# : (콜론)을 붙인다
# 단, 콜론 뒤에는 반드시 반환할 값이 있어야 한다
# 왜냐하면 표현식(expression)은 반드시 값으로 평가되어야 하기 때문이다

print((lambda : 1)()) # 1 

x = 10
print((lambda :x)()) # 10


print("======================================================================================================")
#############################################################################################################

# lambda, map, reduce, filter


print("======================================================================================================")
#############################################################################################################
# map(함수, 리스트)
# 이 함수는 함수와 "리스트" 를 인자로 받는다
# 리스트로부터 원소를 하나씩 꺼내서 함수를 적용시킨 다음, 그 결과를 새로운 리스트에 담아준다


print(list(map(lambda x: x**2, range(5)))) # [0, 1, 4, 9, 16]
# 위의 map 함수가 인자로 받은 함수는 lambda x: x**2 이고, list로는 range(5)를 받았다
# range(5) : [0, 1, 2, 3, 4] 라는 리스트를 돌려줌

# print(map(lambda x: x**2, range(5))) # <map object at 0x0000020B0E9F8B08>

# 위의 예제를 람다가 아닌 보통의 함수로 구현은 어떻게? 직접 해 보시오

# 직접 만들어 본 함수(처음 스스로 만들어 봤다)

def square(x) :
    squ_list = []
    for i in range(x) :
        i = i**2
        squ_list.append(i)
    return squ_list

print(square(10))

# print(list(map(square, [1, 2, 3]))) 


print("======================================================================================================")
#############################################################################################################
# reduce()
# reduce(함수, 순서형 자료)
# 순서형 자료(문자열, 리스트, 튜플)의 원소들을 누적시키면서 함수에 적용

from functools import reduce # reduce는 원래 내장함수였는데 python3 부터 내장함수에서 빠졌다고 한다
reduce(lambda x, y : x+y, [0, 1, 2, 3, 4])

print(reduce(lambda x, y : x+y, [0, 1, 2, 3, 4])) # 10
# 0 + 1을 더하고 그 결과에 +2, 또 그 결과에 + 3, 또 그 결과에 + 4

# what about this?
print(reduce(lambda x, y : y+x, 'abcde')) # edcba why? -> x + y가 아닌 y + x 이기 때문에




print("======================================================================================================")
#############################################################################################################
# filter
# filter(함수, 리스트)
# 리스트에 있는 원소들을 함수에 적용시켜서 결과가 참인 값들로만 새로운 리스트를 만들어준다
# 아하, 조건에 맞는 것들만 출력하는, 필터링을 거쳐서 나오는 느낌

# 다음은 0부터 9까지의 리스트 중에서 5보다 작은 것들만 돌려주는 예제이다

# filter(lambda x: x<5, range(10)) # 파이썬2
list(filter(lambda x:x <5, range(10))) # 파이썬2 및 파이썬3 (파이썬3에서는 list로 꼭 묶어줘야만 하는구먼,,)
                                       # reduce 처럼 따로 from~ import 하지 않으니까 파이썬 내장함수임을 알 수 있다

print(list(filter(lambda x: x<5, range(10)))) # [0, 1, 2, 3, 4]

# 홀수만 돌려주는 filter를 만들어보자

print(list(filter(lambda x : x%2, range(10)))) # [1, 3, 5, 7, 9] # 성공




print("======================================================================================================")
#############################################################################################################
# 보충


# 람다 표현식에 조건부 표현식 사용하기

# lambda 매개변수들: 식1 if 조건식 else 식2

# 다음은 map을 사용하여 리스트 a에서 3의 배수를 문자열로 변환한다

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(map(lambda x: str(x) if x % 3 == 0 else x, a))) # [1, 2, '3', 4, 5, '6', 7, 8, '9', 10]

# map은 리스트의 요소를 각각 처리하므로 lambda의 반환값도 요소여야 한다
# 여기서는 요소가 3의 배수일 때는 str(x)로 요소를 문자열로 만들어서 반환했고
# 3의 배수가 아닐 떄는 x로 요소를 그대로 반환했다

# 람다 표현식 안에서 조건부 표현식 if, else를 사용할 때는 :(콜론)을 붙이지 않는다!!!!!!
# 일반적인 if, else와 문법이 다르므로 주의해야 한다
# 조건부 표현식은 식1 if 조건식 else 식2 형식으로 사용하며 식1은 조건식이 참일 때, 식2은 조건식이 거짓일 때 사용할 식이다
# 특히 람다 표현식에서 if를 사용했다면 반드시 else를 사용해야 한다! 신기방기
# 다음과 같이 if만 사용하면 문법 에러가 발생하므로 주의해야 한다

# print(list(map(lambda x: str(x) if x % 3 == 0, a))) # SyntaxError: invalid syntax

# 그리고 lambda 표현식 안에서는 elif를 사용할 수 없다
# 따라서 조건부 표현식은 << 식1 if 조건식1 else 식2 if 조건식2 else 식3 >> 형식처럼
# if를 연속으로 사용해야 한다


# 리스트에서 1은 문자열로 반환하고 2는 실수로 변환 3 이상은 10을 더하는 식을 만들어 보자
a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(map(lambda x: str(x) if x == 1 else float(x) if x == 2 else x+10, a)))
# ['1', 2.0, 13, 14, 15, 16, 17, 18, 19, 20] # 성공

# 별로 복잡하지 않은 조검인데도 불구, 알아보기가 힘들다
# 이런 경우에는 어거지로 람다 쓰기보다는 def로 함수를 만들고 if, elif, else를 사용하는 것이 좋다

def f(x) :
    if x == 1 :
        return str(x)
    elif x == 2 :
        return float(x)
    else :
        return x+10

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(list(map(f, a)))

# class map 설명
# map(func, *iterables) --> map object
# Make an iterator that computes the function using arguments from each of the iterables. Stops when the shortest iterable is exhausted



print("======================================================================================================")
#############################################################################################################

# map에 객체를 여러 개 넣기
# map은 리스트 등의 반복 가능한 객체를 여러 개 넣을 수도 있다( 반복 가능한 객체? iterator와 관련 있는 건가,,)
# 다음의 두 리스트의 요소를 곱해서 새 리스트를 만들자

a = [1, 2, 3, 4, 5]
b = [2, 4, 6, 8, 10]

print(list(map(lambda x, y : x * y, a, b))) # a, b 이렇게도 들어갈 수 있구만
# [2, 8, 18, 32, 50]

# 이렇게 리스트 두 개를 처리할 때는 람다 표현식에서 lambda x, y : x * y 처럼 매개변수 두 개를 지정하면 된다
# 그리고 map에 람다 표현식을 넣고 그 다음에 리스트 두 개를 콤마로 구분해서 넣어준다
# 즉 람다 표현식의 매개변수 개수에 맞게 반복 가능한 객체도 콤마로 구분해서 넣어주면 된다


print("======================================================================================================")
#############################################################################################################

# filter 사용하기
# filter는 반복 가능한 객체에서 특정 조건에 맞는 요소만 가져온다
# filter에 지정한 함수의 반환값이 True일 때만 해당 요소를 가져온다
# filter(함수, 반복 가능한 객체) ############ 반복 가능한 객체는 무엇인가,, Iterator 파일로 가시오(정리 예정)

# 일반 함수 형식
def f(x) :
    return x > 5 and x < 10

a = [8, 3, 2, 10, 15, 7, 1, 9, 0, 11]
print(list(filter(f, a))) # [8, 7, 9]

# lambda 형식
print(list(filter(lambda x: x>5 and x<10, a))) # [8, 7, 9]




print("======================================================================================================")
#############################################################################################################

# reduce 사용하기
# reduce는 반복 가능한 객체의 각 요소를 지정된 함수로 처리한 뒤 이전 결과와 누적해서 반환하는 함수

# 일반 함수 형식
def f(x, y) :
    return x + y

a = [1, 2, 3, 4, 5]

from functools import reduce
print(reduce(f, a)) # 15

# lambda 형식
print(reduce(lambda x, y : x + y, a))



#############################################################################################################
# 참고
# map, filter, reduce와 리스트 표현식
# 리스트(딕셔너리, 세트) 표현식으로 처리 할 수 있는 경우에는 map, filter와 람다 표현식 대신
# 리스트 표현식으로 사용하는 것이 좋다
# list(filter(lambda x: x>5 and x<10, a))는 다음과 같이 리스트 표현식으로도 만들 수 있다

a = [8, 3, 2, 10, 15, 7, 1, 9, 0, 11]

print([i for i in a if i > 5 and i < 10]) # [8, 7, 9]
# 위와 같은 리스트 표현식이 좀 더 알아보기 쉽고 속도도 더 빠르다

# 또한, for, while 반복문으로 처리할 수 있는 경우에도 reduce 대신 for, while을 사용하는 것이 좋다
# 왜냐하면 reduce는 코드가 조금만 복잡해져도 의미하는 바를 한 눈에 알아보기가 힘들기 때문
# 이러한 이유로 파이썬 3부터 reduce가 내장 함수에서 제외되었다(그래서 그랬구나....)

# reduce(lambda x, y : x + y, a)는 다음과 같이 for 반복문으로 표현할 수 있다
a = [1, 2, 3, 4, 5]
x = a[0]

for i in range(len(a)- 1) :
    x = x + a[i + 1]
    
print(x)
 # 15


# 처음에 print 문을 for문 범위 안에 적어뒀더니 한 바퀴 돌 때마다 나온 값을 모두 출력
# 마지막 딱 최종값만 출력하고 싶다면 for문 영역에서 빠져 나와서 print문을 써야 한다
a = [1, 2, 3, 4, 5]
x = a[0]


for i in range(len(a)- 1) :
    x = x + a[i + 1]
    print(x) 
# 3
# 6
# 10
# 15