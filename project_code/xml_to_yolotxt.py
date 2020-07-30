import glob, os, shutil
from xml.etree.ElementTree import parse
 
labels = ['kicking', 'punching', 'pushing']
labelpath = 'C:/darknet-master/darknet-master/build/darknet/x64/mydata/annotations/'
imgpath = 'C:/darknet-master/darknet-master/build/darknet/x64/mydata/images/'
 
def file_path_save(): 
    filenames = [] 
    files = os.listdir(labelpath)
    
    for file in files:
        xmlfile = parse(labelpath + file)
        if xmlfile.getroot().find('object') is None:
            print(imgpath + file[:-3] +'jpg')
            print(labelpath + file)
            os.remove(imgpath + file[:-3] +'jpg')
            os.remove(labelpath + file)
            continue
 
        tmp = xmlfile.getroot().find('object').find('name').text
        width = int(xmlfile.getroot().find('size').find('width').text)
        height = int(xmlfile.getroot().find('size').find('height').text)
        xmin = int(xmlfile.getroot().find('object').find('bndbox').find('xmin').text)
        ymin = int(xmlfile.getroot().find('object').find('bndbox').find('ymin').text)
        xmax = int(xmlfile.getroot().find('object').find('bndbox').find('xmax').text)
        ymax = int(xmlfile.getroot().find('object').find('bndbox').find('ymax').text)
        with open(imgpath+ file[:-3] +'txt', 'a') as f:
            f.write(str(labels.index(tmp)) + ' ' + str((xmax+xmin)/(2*width)) + ' '+ str((ymax+ymin)/(2*height))+ ' ' + str((xmax-xmin)/width) + ' '+ str((ymax-ymin)/height) +'\n')
    # for i in range(len(train)):
    #     with open("C:/Users/bitcamp/Downloads/darknet-master/build/darknet/x64/mydata/train.txt", 'a') as f:
    #         f.write(train[i] + "\n") 
    # for i in range(len(val)):
    #     with open("C:/Users/bitcamp/Downloads/darknet-master/build/darknet/x64/mydata/val.txt", 'a') as f:
    #         f.write(val[i] + "\n") 
 
if __name__ == '__main__': file_path_save()
 
