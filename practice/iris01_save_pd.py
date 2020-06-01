'''
import numpy as np # numpy는 항상 필요
import pandas as pd # pandas를 불러오고 pd라 부르겠다

datasets = pd.read_csv("./data/csv/iris.csv",
                        index_col = None,
                        header = 0, sep = ',')

                        # pandas 파일을 보면 comma로 데이터를 나누어놓음
                        # comma를 sep, 빼 주면 딱, 숫자로 이루어진 데이터가 된다
                        # index_col = None :  이 데이터에는 따로 index column이 없음을 명시하는 것
                        # header = 0 : 첫 번째 행(index[0])은 헤더로 빠진다
                        # 데이터에 따라 다르니까 그 때 그 때 보고 판단해야 함
                        # header = None 해 버리면 이 데이터에서 header는 존재하지 않고
                        # 그냥 데이터가 통으로 실제 데이터가 되어버린다
                        # datasets은 그냥 변수명
                        # 현재 작업 폴더의 하위 폴더 data - csv 안에 있는 iris.csv라는 파일을
                        # 불러들여서 읽고 datasets라는 변수에 넣는 것
                        # 데이터 불러오기 완료

print(datasets) # 를 해 보면 자동으로 index를 생성시켜 준다
                # print가 잘 되면 정상적으로 데이터를 부른 것

print(datasets.head()) # 전체 데이터에서 위에서부터 5개를 보여준다

print(datset.tail()) # 전체 데이터에서 아래에서부터 5개를 보여준다

# index와 header에 주의해야 한다, 데이터 분석하는 것은 많은 연습, 실습이 필요할 것 같다

print(datasets.values) # datasets에서 values만 보여준다? 
                       # machine을 돌리기 위해 pandas를 numpy 형태로 바꿔준다
                       # machine이 알아보기 쉽도록?

a = datasets.values # a라는 변수에 datasets.values를 대입

print(type(a)) # a 즉, datasets.values 의 형태를 확인
               # <class 'numpy.ndarray> 가 출력
               # numpy의 ndarry 형태이다 정도로 알면 될 듯하다

np.save('./data/iris.npy', arr = a)
# a 즉, datasets.values를 data 폴더의 iris.npy 라는 이름으로 저장한다
# 데이터 저장이 완료 되었다
# 모델 구성 시 저장 된 이 데이터를 load 해서 모델 구성을 하면 된다!!!!
# 위치 표시 할 때 '.'을 까먹지 말자
# 현재 작업하는 가장 큰? 상위폴더, 현 위치를 말하는 것(여기서는 deepstudy가 되겠지)




'''