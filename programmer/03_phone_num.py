# 핸드폰 뒷 4자리를 제외하고 나머지 숫자를 *로 가리기

ph_1 = "01033334444"
ph_2 = "027778888"

# print(ph_2[:-4])

# print(len(ph_1[:-4]))
# print(len(ph_2[:-4]))

def solution(phone_number):
    a = "*" * len(phone_number[:-4])
    answer = phone_number.replace(phone_number[:-4], a)
    return answer

print(solution(ph_1))
print(solution(ph_2))


# 2줄로 끝낸 ㅋㅋㅋ
# 다른 사람들의 답안
def hide_numbers(s) :
    return "*" * (len(s)-4) + s[-4:]

print(hide_numbers(ph_1))