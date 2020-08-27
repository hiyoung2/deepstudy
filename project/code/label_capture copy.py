import cv2

vidcap = cv2.VideoCapture('D:/watcher_data/data/videos/inside_croki_03/411-2/411-2_cam01_assault01_place08_night_summer.mp4')

count = 0

while(vidcap.isOpened()):
    ret, image = vidcap.read()

    if ret is False :
        print("**************")
        continue
    else :
        cv2.imwrite('D:/watcher_data/cnn/normal/normal%d.jpg' % count, image)

        print('Saved frame%d.jpg' % count)

        count += 1

vidcap.release()

print("캡쳐완료")

