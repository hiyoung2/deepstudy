# os : 파이썬 모듈 함수

# 데이터 사이언스에 가장 애용되는 언어 중 하나인 python
# by 귀도 반 로썸, 프로그래밍을 재밌게 만들고자 하는 취지 하에 개발한 언어로, 누구든 쉽게 배울 수 있다

# os = operating system의 약자로, 운영체제를 의미
# 파이썬에 기본적으로 내장된 모듈의 이름
# 이 os 모듈은 운영체제와의 상호작용을 돕는 다양ㅇ한 기능을 제공

# 3가지 기능
# 1. 현재 디렉토리 확인
# 디렉토리는 비유하자면 파일이 정리된 서랍
# 파이썬이나 다른 언어에서 어떤 파일을 끌어다 불러오고 싶을 때,
# 그냥 파일명을 부르는 게 아니라
# "무슨 드라이브의 무슨 유저의 파일 서랍의 바탕화면에 있는 무슨 파일을 가져와"
# 라고 구체적으로 말해줘야 한다
# 이런 기능을 수행하기 위해 os 모듈을 호출
# import os 라는 코드를 통해 os 모듈을 가져올 수 있다

import os
os.getcwd() # PS C:\hiyoung\deepstudy>
# "os.": "os 모듈의~"
# getcwd() 라는 함수를 사용, 이 코드를 실행하면 위와 같은 경로가 나온다
# get the path of current working directory라는 의미

# 2. 디렉토리 변경
os.chdir("C:\hiyoung\deepstudy\codingking\codeup")
# chdir : change working directory

# 3. 현재 디렉토리의 파일 목록 확인하기
# 파일을 불러오기 전에 어떤 파일 목록이 있는지 확인해보고싶다면 
# listdir() 함수를 사용
os.listdir
print(os.listdir())
# ['1001.py', '1002.py', '1003.py', '1004.py', '1005.py', '1006.py', '1007.py', '1010.py'...]