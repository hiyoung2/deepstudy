# 14.4.1 키별 통계량 산출

# 14.1.1절 'Pandas로 CSV 읽기'에서 사용했던 와인의 데이터셋을 다시 사용하여 열의 평균값을 산출하자

import pandas as pd

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header = None)
df.columns = ["", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", 
             "Nonflavanoid phenols", "Proanthocyanis", "Color intensity", "Hue", "0D280/0D315 of diluted wines", "Proline"]


print(df["Alcohol"].mean())