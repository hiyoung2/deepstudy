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

# 아래 Person 이라는 Class는 init, work, sleep 3개의 메서드를 가지고 있다
# init은 특히나 파이썬 클래스가 반드시 가져야 하는 메서드!

class Person:
    def __init__(self, name):
        self.name = name
        print(self.name + " is initialized")
    
    def work(self, company):
        print(self.name + " is workin in " + company)
    
    def sleep(self):
        print(self.name + " is sleeping")


obj = Person("PARK") 
# Person Class를 인스턴스 obj를 이용하여 생성
# 생성자인 init 내에서 입력으로 받은 name(PARK)을 self.name에 대입을 한다
# self.name은 Class의 멤버 변수를 가리킨다


# method call
obj.work("ABCDEF")
obj.sleep()

# 속성에 직접 접근, 기본적으로 파이썬에서는 모두 public
print("current person object is ", obj.name)
# 여기서 name은 Person Class가 가지고 있는 내부의 인스턴스 변수인데
# 인스턴스 obj를 이용해 바로 값을 가져올 수 있다
# 파이썬에서는 메서드와 멤버변수 모두가 기본적으로 public으로 선언되기 때문에
# 외부에서 생성된 인스턴스를 통해 바로 접근하여 사용가능하다
