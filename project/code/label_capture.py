import cv2
import os,shutil
from xml.etree.ElementTree import parse

main_file_path = 'D:/watcher/videos/videos'
main_save_path = 'D:/watcher/videos/images'

count = 0

for m in os.listdir(main_file_path):
    for n in os.listdir(main_file_path + '/' + m):
        file_path = main_file_path + '/' + m + '/' + n
        file_list = os.listdir(file_path) 

        print(file_list)
        for i in file_list :
            if i[-3:] != 'mp4':
                continue
            path = file_path + '/' + i

            if os.path.isdir(main_save_path + '/' + i[:-4]):
                shutil.rmtree(main_save_path + '/' + i[:-4])
            os.mkdir(main_save_path +'/' + i[:-4])

            file_xml = i[:-3] + 'xml'
            tree = parse(file_path + '/' + file_xml)
            for action in tree.getroot().find("object").findall("action"):
                action_name = action.find("actionname").text

                if os.path.isdir(main_save_path + '/' + i[:-4] +'/' + action_name):
                    shutil.rmtree(main_save_path + '/' + i[:-4]+'/' + action_name)
                os.mkdir(main_save_path +'/' + i[:-4]+'/' + action_name)
                print(action.findall("frame")[0].find("start").text)
                for frame in action.findall("frame"):
                    vidcap = cv2.VideoCapture(path)

                    start = int(frame.find("start").text)
                    end = int(frame.find("end").text)

                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)

                    ret = True
                    while(ret) :
                        ret, image = vidcap.read() 
                        now = int(vidcap.get(1))
                        if ret is False :
                            print("**************")
                            continue
                        if(now > end) :
                            break
                        if(now % 7 == 0) : 
                            print('Saved frame number :' + str(int(vidcap.get(1))))
                            cv2.imwrite(main_save_path + '/' + i[:-4] + '/' + action_name +'/' + action_name + '_frames%d.jpg' % count, image) 
                            print('Saved frame%d.jpg' % count)
                            count += 1
                    vidcap.release()

print("캡쳐 완료")