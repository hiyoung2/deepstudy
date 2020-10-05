import sys
import cv2

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('./opencv/vtest.avi')

if not cap.isOpened():
    print('camerea open failed')
    sys.exit() 

while True:
    ret, frame = cap.read()

    if not ret: # if frame is None:으로 써도 된다
        break

    edge = cv2.Canny(frame, 50, 150) # edge detection 
                                     # frame, 50, 150 == parameters to detect "edge"

    cv2.imshow('frame', frame)
    cv2.imshow('edge', edge)

    if cv2.waitKey(20) == 27: # ESC key
        break

cap.release()
cv2.destroyAllWindows()
