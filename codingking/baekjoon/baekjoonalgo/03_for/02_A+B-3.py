# --title--
# 10950번: A+B - 3

# --problem_description--
# 두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.


# --problem_input--
# 첫째 줄에 테스트 케이스의 개수 T가 주어진다.

# 각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)


# --problem_output--
# 각 테스트 케이스마다 A+B를 출력한다.

t = int(input())

# 1차 
# for a, b in range(t) :
#     a, b = map(int, input().split())
#     print(a+b)
# TypeError: cannot unpack non-iterable int object

# 2차
# for i in range(t) :
#     a, b = map(int, input().split())
#     print(a + b)
'''
5
1 1
2
2 2
4
4 4
8
5 5
10
6 6
12
'''
# 3차
# for i in range(t) :
    # a, b = map(int, input().split())

cnt = 5
# for i in range(cnt) :
    
    