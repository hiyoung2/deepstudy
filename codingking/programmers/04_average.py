# 배열의 평균값 구하는 함수

import numpy as np

def solution(arr):
    arr = np.array(arr)
    answer = np.mean(arr)
    return answer


# 다른 사람 풀이
# 1) 한 줄로 끝냄 ㅎㅎ
def average_1(list) :
    return (sum(list) / len(list))


list = [5, 3, 4]
# print("평균값 : {}".format(average(list)))
print("평균값 : %.2f" % average_1(list))

# 2) 분모가 0일 경우를 생각
# (주어진 문제에서는 분모 0인 경우를 배제하였으나,,)
# ZeroDvisionError 예외를 피할 수 있다


def average_2(list) :
    if len(list) == 0 :
        return 0
    
    return sum(list) / len(list)

list = [5, 3, 4]
print("평균값 : {}".format(average_2(list)))