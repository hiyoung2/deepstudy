# 가변인자

# def profile(name, age, lang1, lang2, lang3, lang4, lang5):
#     print("이름 : {0}\t나이 : {1}\t".format(name, age), end=" ")  # end=" " : 줄바꿈 하지 않고 그대로 이어서 출력
#     print(lang1, lang2, lang3, lang4, lang5)

# profile("하사장", 20, "Python", "Java", "C", "C++", "C#")
# profile("김사장", 25, "Kotlin", "Swift", "", "", "")

# 그런데, 하사장이 언어 하나를 더 할 수 있게 되거나 김사장도 다른 언어를 추가해야 할 수 있음
# 그러면 함수 자체를 lang6, 7, ... 바꿔줘야 함.
# 이 때 가변인자를 사용하면 해결 가능

def profile(name, age, *language):
    print("이름 : {0}\t나이 : {1}\t".format(name, age), end=" ")
    for lang in language:
        print(lang, end=" ")
    print()

profile("하사장", 20, "Python", "Java", "C", "C++", "C#", "JavaScript")
profile("김사장", 25, "Kotlin", "Swift")
