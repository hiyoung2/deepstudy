import sys

mk = list(map(int, sys.stdin.readline().split()))
ms = list(map(int, sys.stdin.readline().split()))

mk_total = sum(mk)
ms_total = sum(ms)

if mk_total > ms_total :
    print(mk_total)
elif mk_total < ms_total :
    print(ms_total)
else :
    print(mk_total)