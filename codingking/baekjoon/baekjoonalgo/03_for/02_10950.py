# --title--
# 10950번: A+B - 3

# --problem_description--
# 두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.


# --problem_input--
# 첫째 줄에 테스트 케이스의 개수 T가 주어진다.

# 각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)


# --problem_output--
# 각 테스트 케이스마다 A+B를 출력한다.

# 1차 : 출력형태가 틀림
t = int(input())
for i in range(t) :
    a, b = map(int, input().split())
    print(a + b)
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
# 2차 
# 1. 테스트 케이스의 수를 입력 받아야 함
# 2. 테스트 케이스에 들어갈 두 숫자를 입력 받아야 함
# 3. 케이스마다 더한 수들을 하나씩 출력해야 하는데 출력형태를 보니 for문을 실행해야 할 듯
# 4. 더한 수들을 하나씩 꺼내기 전에 일단 리스트로 묶어 두고 인덱스로 하나씩 뽑아내면 될 듯하다

cnt = int(input()) # 1
res = [] # 4를 위해 빈 리스트 하나 생성
for i in range(cnt) : # 테스트 케이스 수만큼 for문 실행
    a, b = map(int, input().split()) # 두 수를 입력 받는다
    cnt -= 1 # 테스트 케이스 수가 하나씩 줄어들면서 cnt 에 저장된 값만큼 for문이 실행
    total = a + b # 최종 출력을 위한 두 수의 합을 total 변수에 둠
    res.append(total) # 더한 수들을 res 라는 리스트에 append

for j in range(len(res)) : # res 리스트의 요소 수만큼 for문 실행
    print(res[j]) # 리스트 res에서 요소 하나하나씩 출력
    
######################################################################
# 여러 수를 입력 받을 때 input() 대신
# sys.stdin.readline()으로 하면 더 빠른 속도로 처리 가능하다

