# 기본값

# def profile(name, age, main_lang):
#     print("이름 : {0}\t나이 : {1}\t주 사용 언어 : {2}"\
#         .format(name, age, main_lang))

# profile("하사장", 20, "파이썬")
# profile("김사장", 25, "자바")



# 같은 학교, 같은 학년, 같은 반, 같은 수업
def profile(name, age=17, main_lang="파이썬"): # 나이, 주사용언어에는 기본값이 설정
    print("이름 : {0}\t나이 : {1}\t주 사용 언어 : {2}"\
        .format(name, age, main_lang))

profile("하사장")
profile("김사장")