# 클래스, Class

# 클래스 변수(class variable)는 해당 클래스로 생성된 모든 인스턴스가 공통으로 사용하는 변수
# -> 클래스 변수는 클래스 내외부에서 "클래스명.클래스변수명"으로 접근할 수 있음

# 클래스 메서드(class method)는 메서드 앞에 #classmdthod를 반드시 표시하여 해당 메서드가 클래스 메서드임을 표시함
# -> 클래스 메서드는 객체 인스턴스를 의미하는 self 대신 cls라는 클래스를 의미하는 파라미터를 인수로 전달받음

class Person:
    
    count = 0  # class variable

    def __init__(self, name):
        self.name = name
        Person.count += 1 # class 변수 count 증가
        print(self.name + " is initialized")

    def work(self, company):
        print(self.name + " is working in " + company)

    def sleep(self):
        print(self.name + " is sleeping")

    @classmethod
    def getCount(cls): # class method
        return cls.count


# Person instance 2개 생성
obj1 = Person("PARK")
obj2 = Person("KIM")

# method call
obj1.work("ABCDEF")

obj2.sleep()

# 속성에 직접 접근, 기본적으로 파이썬에서는 모두 public
print("current person object is ", obj1.name, ", ", obj2.name)

# class variable direct access
print(Person.count)
