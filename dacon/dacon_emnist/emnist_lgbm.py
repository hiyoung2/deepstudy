# import library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts
from sklearn.decomposition import PCA

import time
import random
import datetime

# load data

train = pd.read_csv("./dacon/dacon_emnist/data/train.csv")
test = pd.read_csv("./dacon/dacon_emnist/data/test.csv")
submission = pd.read_csv("./dacon/dacon_emnist/data/submission.csv")

# data 구성
# 본 문제의 목표는 기존의 mnist와 다르게 문자 속에 숨어 있는 숫자를 예측하는 것

train.set_index('id').head()