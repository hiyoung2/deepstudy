# darknet에서 train에 사용할 이미지 경로 텍스트 파일 만들기

import os
import sys

# main_file_path = 'D:/watcher/videos/videos'
# main_save_path = 'D:/watcher/videos/images'

# count = 0

# for m in os.listdir(main_file_path):
#     for n in os.listdir(main_file_path + '/' + m):
#         file_path = main_file_path + '/' + m + '/' + n
#         file_list = os.listdir(file_path) 

#         print(file_list)

sys.stdout = open('darknettest.txt', 'a')
open = ('./darknettest.txt', 'a')

main_file_path = 'C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/mydata'

for m in os.listdir(main_file_path) :
    if m[-3:] == 'jpg' :
        print('mydata/testdata/' + m)