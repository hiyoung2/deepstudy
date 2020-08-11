# import numpy as np
# import cv2
# import pyyolo
# cap = cv2.VideoCapture('D:/watcher/videos/videos/inside_croki_03/407-2')
# meta_filepath = "C:/darknet-master/darknet-master/build\darknet/x64/mydata/testdata.data"
# cfg_filepath = "C:/darknet-master/darknet-master/build\darknet/x64/mydata/yolov3-testdata.cfg"
# weights_filepath = "C:/darknet-master/darknet-master/build\darknet/x64/mydata/backup/yolov3-testdata_final.weights"



# meta = pyyolo.load_meta(meta_filepath)
# net = pyyolo.load_net(cfg_filepath, weights_filepath, False)


# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if not ret:
#         break
#     yolo_img = pyyolo.array_to_image(frame)
#     res = pyyolo.detect(net, meta, yolo_img)
#     for r in res:
#         cv2.rectangle(frame, r.bbox.get_point(pyyolo.BBox.Location.TOP_LEFT, is_int=True),
#                       r.bbox.get_point(pyyolo.BBox.Location.BOTTOM_RIGHT, is_int=True), (0, 255, 0), 2)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

##########################################################################################################################

from darknet import darknet

print("import complete")

import cv2, numpy as np
from ctypes import *
 
net = darknet.load_net(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/yolov3.cfg", b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/yolov3.weights", 0) 
meta = darknet.load_meta(b"C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/data/coco.data") 
cap = cv2.VideoCapture("C:/Users/bitcamp/anaconda3/Lib/site-packages/darknet/mydata/6-1_cam01_assault01_place03_night_spring.mp4") 
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while(cap.isOpened()):
    ret, image = cap.read()
    image = cv2.resize(image, dsize=(480, 640), interpolation=cv2.INTER_AREA)
    print(image.shape)
    if not ret: 
        break 

    frame = darknet.nparray_to_image(image)
    r = darknet.detect_image(net, meta, frame) 
 
    boxes = [] 
 
    for k in range(len(r)): 
        width = r[k][2][2] 
        height = r[k][2][3] 
        center_x = r[k][2][0] 
        center_y = r[k][2][1] 
        bottomLeft_x = center_x - (width / 2) 
        bottomLeft_y = center_y - (height / 2) 
        x, y, w, h = bottomLeft_x, bottomLeft_y, width, height 
        boxes.append((x, y, w, h))
 
    for k in range(len(boxes)): 
        x, y, w, h = boxes[k] 
        top = max(0, np.floor(x + 0.5).astype(int)) 
        left = max(0, np.floor(y + 0.5).astype(int)) 
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int)) 
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int)) 
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2) 
        cv2.line(image, (top + int(w / 2), left), (top + int(w / 2), left + int(h)), (0,255,0), 3) 
        cv2.line(image, (top, left + int(h / 2)), (top + int(w), left + int(h / 2)), (0,255,0), 3) 
        cv2.circle(image, (top + int(w / 2), left + int(h / 2)), 2, tuple((0,0,255)), 5)
 
    cv2.imshow('frame', image) 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
 
cap.release()
