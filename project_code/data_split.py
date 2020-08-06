# import random

# txt = open('D:/darknet-master/build/darknet/x64/data/data/train.txt','r')
# f = open('D:/darknet-master/build/darknet/x64/data/data/train_suffle.txt','w')

# tmp = []

# while True :
#     line = txt.readline()
#     if not line:
#         break
        
#     tmp.append(line)
    
# random.shuffle(tmp)
        
# for i in tmp :  
#     f.write(i)

# txt.close()
# f.close()

import glob 
from sklearn.model_selection import train_test_split
def file_path_save(): 
    trainlist = []
    vallist = []

    files = sorted(glob.glob("C:/darknet-master/darknet-master/build/darknet/x64/mydata/testdata/*.jpg")) 
    train_files, val_files = train_test_split(
        files, shuffle=True, train_size=0.8, random_state=66
    )
    print(len(train_files))
    print(len(val_files))

    trainlist.append(train_files)
    vallist.append(val_files)

    print(trainlist)
    print("========================================================")
    print(vallist)

    # for i in range(len(train_files)): 
    #     f = open("C:/darknet-master/darknet-master/build/darknet/x64/mydata/train.txt", 'a') 
    #     f.write(train_files[i] + "\n") 

    for i in range(len(val_files)): 
        f = open("C:/darknet-master/darknet-master/build/darknet/x64/mydata/val.txt", 'a') 
        f.write(val_files[i] + "\n") 

if __name__ == '__main__': file_path_save()