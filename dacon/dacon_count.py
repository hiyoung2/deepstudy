import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.models import Model
from keras.layers import Dense, Dropout, Input

train = pd.read_csv("./data/csv/train.csv", header = 0, index_col = 0)

test = pd.read_csv("./data/csv/test.csv", header = 0, index_col = 0)

submission = pd.read_csv("./data/csv/sample_submission.csv", header = 0, index_col = 0)

train.groupby('na')['na'].count()

count_data = train.groupby('na')['na'].count()



count_data.plot()
plt.show()