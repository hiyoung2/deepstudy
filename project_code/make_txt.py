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

sys.stdout = open('testdata.txt', 'a')
open = ('./testdata.txt', 'a')

main_file_path = 'C:/darknet-master/darknet-master/build/darknet/x64/mydata/testdata'

for m in os.listdir(main_file_path) :
    if m[-3:] == 'jpg' :
        print('mydata/testdata/' + m)