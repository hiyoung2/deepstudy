# 짝수 홀수 가리기 문제

def solution(num):
    if(num %2 == 0) :
        answer = "Even"
    else :
        answer = "Odd"
    return answer

print(solution(4))
