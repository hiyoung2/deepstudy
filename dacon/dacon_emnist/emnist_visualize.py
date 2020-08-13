import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('pandas',pd.__version__) # 1.0.5
print('numpy',np.__version__) # 1.18.5

# TRAIN
train = pd.read_csv('./dacon/dacon_emnist/data/train.csv')
train_digit = train['digit'].values # 숨어 있는 숫자, digit 값만 불러와서 변수에 저장
train_letter = train['letter'].values # 문자 
train_img = train.iloc[:,3:].values.reshape(-1, 28, 28, 1).astype(np.int)

# TEST
test = pd.read_csv('./dacon/dacon_emnist/data/test.csv')
test_img = test.iloc[:,2:].values.reshape(-1, 28, 28, 1).astype(np.int)
X_test_letter = test['letter'].values

del train, test

print('digit  : ', np.unique(train_digit))
print('letter : ', np.unique(train_letter))

# digit  :  [0 1 2 3 4 5 6 7 8 9]
# letter :  ['A' 'B' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'J' 'K' 'L' 'M' 'N' 'O' 'P' 'Q' 'R' 'S' 'T' 'U' 'V' 'W' 'X' 'Y' 'Z']


for idx in range(100):
    plt.figure(figsize=(20,30))
    #####################################################################################################
    plt.subplot(1,9,1)
    plt.imshow(train_img[idx].reshape(28,28),cmap='gray')
    plt.axis('off')
    
    plt.title('digit:{}   letter:{}'.format(train_digit[idx], train_letter[idx]), loc='left', fontsize=20)

    plt.subplot(1,9,2)
    data = np.where(train_img>=150, train_img, 0)
    plt.imshow(data[idx].reshape(28,28),cmap='gray')
    plt.axis('off')

    plt.subplot(1,9,3)
    plt.imshow(np.zeros((28,28,3))+1,cmap='gray')
    plt.axis('off')
    
    #####################################################################################################
    plt.subplot(1,9,4)
    plt.imshow(train_img[idx+1].reshape(28,28),cmap='gray')
    plt.axis('off')
    
    plt.title('digit:{}   letter:{}'.format(train_digit[idx+1], train_letter[idx+1]), loc='left', fontsize=20)

    plt.subplot(1,9,5)
    data = np.where(train_img>=150, train_img, 0)
    plt.imshow(data[idx+1].reshape(28,28),cmap='gray')
    plt.axis('off')

    plt.subplot(1,9,6)
    plt.imshow(np.zeros((28,28,3))+1,cmap='gray')
    plt.axis('off')
    
    #####################################################################################################
    plt.subplot(1,9,7)
    plt.imshow(train_img[idx+2].reshape(28,28),cmap='gray')
    plt.axis('off')
    
    plt.title('digit:{}   letter:{}'.format(train_digit[idx+2], train_letter[idx+2]), loc='left', fontsize=20)

    plt.subplot(1,9,8)
    data = np.where(train_img>=150, train_img, 0)
    plt.imshow(data[idx+2].reshape(28,28),cmap='gray')
    plt.axis('off')

    plt.subplot(1,9,9)
    plt.imshow(np.zeros((28,28,3))+1,cmap='gray')
    plt.axis('off')
    
plt.show()