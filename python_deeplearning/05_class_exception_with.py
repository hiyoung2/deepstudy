# Class
# 파이썬 클래스는 class 키워드를 사용하여 자신만의 데이터 타입을 만들 수 있다

# class 클래스이름:
#   def __init__(self, 인수, ...): -> 생성자
#   def 메서드이름(self, 인수, ...) -> 메서드

# 파이썬 클래스에서는 __init__ 메ㅔ서드가 생성자(constructor) 역할을 수행하여
# 인스턴스가 만들어질 때 한 번만 호출됨
#
# 파이썬에서는 클래스 메서드의 첫번째 인수로 '자신의 인스턴스'를 나타내는 self를 반드시 기술해야 한다
#
# 기본적으로 파이썬에서는 메서드와 속성 모두 public

class Person:
    def __init__(self, name):
        self.name = name
        print(self.name + " is initialized")
    
    def work(self, company):
        print(self.name + " is workin in " + company)
    
    def sleep(self):
        print(self.name + " is sleeping")


# Person instance 2개 생성
obj = Person("PARK")

# method call
obj.work("ABCDEF")
obj.sleep()

# 속성에 직접 접근, 기본적으로 파이썬에서는 모두 public
print("current person object is ", obj.name)