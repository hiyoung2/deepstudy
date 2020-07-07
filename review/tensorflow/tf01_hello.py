# 텐서플로 impot, tf라고 부름
import tensorflow as tf

# 텐서 플로 버전 확인
print(tf.__version__) # 1.14.0

# 첫 시작은 항상 Hello World 출력

# 이렇게 단순 출력도 되는데 constant를 끌어 들이는 것은 떡밥?
print("Hello World") # Hello World

# 수업 시간 방식
hello = tf.constant("Hello World") # constant : 상수, 변하지 않는 값
                                   # hello에 "Hello World" 라는 상수를 대입
                                   # constant : tensorflow에서 중요한 키워드 같다

# 바로 출력을 해 봄
print(hello) # Tensor("Const:0", shape=(), dtype=string)
# Hello World가 출력 될 것이라 예상 -> 땡
# Tensor 로서(?)의 자료형이 출력
# Constant 이고, shape = () 인 것은 shpae이 따로 없다는 뜻? 데이터타입, 즉 자료형은 string 문자열!!(아는 거 나와서 반가움)

# 예상대로 나오게 하려면?
# 즉, Hello World 라는 문자열을 출력, 보고 싶다면 어떻게 해야 하나?

# tf.Session() 이라는 것을 이용해야 한다!
# 중요! 여기서 텐서플로 1대 버전은 어떻게 작동하는지를 알아야 한다
# 텐서플로는 Session 이라고 하는 쉽게 말해 연산 과정을 거치는 작업이 필수로 이뤄진다
# 따라서 어떤 것을 입력해서 출력하고 싶다면 중간에 Session 과정을 거쳐야 한다
# 불편?
# 이제껏 사용했던 tensorflow 2대 version을 backend로 한 keras에서 작업을 할 때는 전혀 하지 않아도 되는 과정
# 정말 인공지능 이 과정의 첫 시작이 케라스였음에 감사해야 하는 이유
# 이것부터 시작했다면 흠,,

# 그러면 이제 Session을 사용햏보자
sess = tf.Session() # sess라는 이름은 마음대로 지정해도 될 것 같음, 케라스에서 model = ~ 했던 것처럼!
print(sess.run(hello)) # sess만 지정해놓으면 될 것이 아님!
                       # sess.run 이라는 것을 써야 작동이 된다! run : 운영, 작동하다의 의미이므로

# 'Hello World' 이렇게 출력이 되었다
# 사실 b'Hello World' 이렇게 출력이 된 건데, b 는 bytes 라는 뜻을 가지고 있다고 한다

# b가 왜 같이 뜨는지 검색하다가 문자열만 딱 깔끔히 출력되는 방법을 발견
print(sess.run(hello).decode()) # Hello World
# decode는 해석한다는 의미

# The print of string constant is always attaced with 'b' in TensorFlow

# The 'b' prefix is to indicate byte strings rather than unicode strings
# The default depends on your python version: python2 'str' is bytes, but python3 str is unicode
# Briefly : it's a bytes object

# Use sess.run(hello).decode() becuase it is a bytestring
# decode method will return the string
