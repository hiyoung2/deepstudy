class Solution :
    def isPalindrome(self, s:str) -> bool:
        strs = []
        for c in s: # str 안의 character들 모두 조회
            if c.isalnum(): # alphabet or number이면
                strs.append(c.lower())
        # print(strs)
        
        while len(strs) > 1:
            if strs.pop(0) != strs.pop(): 
                return False
            
        return True
                
                
                
                

                
                
                
                
                
                
# pop(0) : array에서 가장 앞의 것을 뽑아 내는 것 : pop이라 하는데, 0을 주면 가장 앞, 0을 안 주면 가장 뒤의 것을 가져온다

# pop()은 리스트의 맨 마지막 요소를 돌려주고 그 요소는 삭제한다.
# 리스트 요소 끄집어내기(pop)

# a 리스트 [1, 2, 3]에서 3을 끄집어내고 최종적으로 [1, 2]만 남는 것을 볼 수 있다.

# pop(x)는 리스트의 x번째 요소를 돌려주고 그 요소는 삭제한다.