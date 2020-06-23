# 행렬 계산 문제
arr1 = [[1, 2], [2, 3]], [[1], [2]]
arr2 = [[3, 4], [5, 6]], [[3], [4]]

import numpy as np

arr1 = np.array(arr1)
print(type(arr1))

def solution(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    answer = arr1 + arr2
    answer.tolist()
    return answer

print(solution(arr1, arr2))
