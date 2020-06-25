# 2739 : 구구단

n = input()
n = int(n)
for i in range(9) :
    i += 1
    print("{0} * {1} = {2}".format(n, i, n*i))
