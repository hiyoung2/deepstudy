
# --url--
# https://www.acmicpc.net/problem/1371

# --title--
# 1371번: 가장 많은 글자

# --problem_description--
# 영어에서는 어떤 글자가 다른 글자보다 많이 쓰인다. 예를 들어, 긴 글에서 약 12.31% 글자는 e이다.

# 어떤 글이 주어졌을 때, 가장 많이 나온 글자를 출력하는 프로그램을 작성하시오.

# --problem_input--
# 첫째 줄부터 글의 문장이 주어진다. 글은 최대 5000글자로 구성되어 있고, 공백, 알파벳 소문자, 엔터로만 이루어져 있다. 그리고 적어도 하나의 알파벳이 있다.

# --problem_output--
# 첫째 줄에 가장 많이 나온 문자를 출력한다. 여러 개일 경우에는 알파벳 순으로 앞서는 것부터 모두 공백없이 출력한다.


# 최빈값이 2개 이상 존재할 때엔 적용이 안 됨
# sentence = list(input())
# print(sentence)
# from collections import Counter
# from statistics import multimode
# cnt = Counter(sentence)
# print(cnt)
# solution = multimode(sentence)
# print(solution)
# print(solution[0][0])
# baekjoon online judge

from collections import Counter
import sys
sentence = sys.stdin.readline()
# 공백, 알파벳 소문자, 엔터로 이루어진
# 여러 문장을 입력받지를 못함

cnt = Counter(sentence)
# print(cnt) # Counter({'e': 3, 'o': 3, 'n': 3, 'j': 2, ' ': 2, 'b': 1, 'a': 1, 'k': 1, 'l': 1, 'i': 1, 'u': 1, 'd': 1, 'g': 1})
# print()
order = cnt.most_common() # 최빈값 순서대로 정렬
# print(order) # [('e', 3), ('o', 3), ('n', 3), ('j', 2), (' ', 2), ('b', 1), ('a', 1), ('k', 1), ('l', 1), ('i', 1), ('u', 1), ('d', 1), ('g', 1)]
# 공백까지 포함해버림
# print()
maximum = order[0][1] # 최빈값이 몇 번 나왔는지

# print(maximum) # 3
# print()

modes = []

for i in order:

    if i [1] == maximum :
        if i [0] == ' ':
            pass
        elif i [0] != ' ':
            modes.append(i[0])

    # if i[1] == maximum:
    #     if i[0] == ' ':
    #         pass
    #     modes.append(i[0])

# print(modes)
modes.sort()
print(''.join(modes))
