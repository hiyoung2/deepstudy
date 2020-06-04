# 머신러닝 맛보기

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
# 0부터 10까지 0.1씩 증가하는 데이터세트

y = np.sin(x) 
# 0.1에 대한 sin값, 0.2에 대한 sin값, ...

plt.plot(x, y)

plt.show()
# 아주 예쁜 sin 함수 그래프가 생성된다