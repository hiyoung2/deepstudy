# if

# weather = "맑아요"
# if weather == "비":           - weather = "비"이면 5의 문장이 출력
#     print("우산을 챙기세요.")
# elif weather == "미세먼지":   - weather = "미세먼지"이면 7의 문장이 출력
#     print("마스크를 챙기세요.")
# else:                        - weather = 비도 미세먼지도 아니면 9의 문장이 출력
#     print("준비물 필요 없어요.") 

# weather = input("오늘 날씨는 어때요? ") # 어때요? 다음에 커서 움직이면서 입력이 가능해짐, str 형태로 weather라는 변수에 저장됨.
# if weather == "비" or weather == "눈": # 비 또는 눈이 입력되면 13의 문장이 출력
#     print("우산을 챙기세요.")
# elif weather == "미세먼지":
#     print("마스크를 챙기세요.")
# else:
#     print("준비물 필요 없어요.")

temp = int(input("기온은 어때요? ")) # 기온은 숫자. input은 str 형태로 받기 때문에 int형으로 감싸줌
if 30 <= temp:
    print("너무 더워요. 나가지 마세요.")
elif 10 <= temp and temp < 30:
    print("괜찮은 날씨에요.")
elif 0 <= temp < 10:    # and 생략하고 이렇게 쓸 수도 있음
    print("외투를 챙기세요.")
else:
    print("너무 추워요. 나가지 마세요.")
