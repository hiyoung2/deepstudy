
import os, shutil

data_path = 'D:/python_module/darknet-master/build/darknet/x64/project/data'
classes = ['fighting', 'normal']
flag = 1

for data in os.listdir(data_path):
    if data[-3:] != 'jpg':
        continue
    print(data)
    with open(data_path + '/' + data[:-3] + 'txt', 'r') as f:
        flag = 1
        for line in f.readlines():
            if line[0] == '0':
                flag = 0
    shutil.copy(data_path + '/' + data, '경로생략.../flow_from/' + classes[flag])
    
    
    
    
    
    
    
    # shutil.copy(data_path + '/' + data, 'D:/python_module/darknet-master/build/darknet/x64/project/flow_from/' + classes[flag])