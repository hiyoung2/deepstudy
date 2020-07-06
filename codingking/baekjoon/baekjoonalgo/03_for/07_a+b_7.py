import sys
cnt = int(sys.stdin.readline())
res = [] 
for i in range(cnt) :
    a, b = map(int, sys.stdin.readline().split()) 
    cnt -= 1 
    total = a + b 
    res.append(total) 

x = 0
for j in range(len(res)) : 
    x += 1
    print("Case #",x,": " + str(res[j]), sep = "")