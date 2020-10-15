# save video file

# 카메라 입력 프레임에서 에지 영상을 구하여 동영상으로 저장하기

import sys, cv2

# 카메라로부터 cv2.VideoCapture 객체 생성
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

# 동영상 저장을 위한 cv2.VideoWriter 객체 생성
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # *'DIVS' == 'D', 'I', 'V', 'X'
out = cv2.VideoWriter('./opencv/output.avi', fourcc, fps, (w, h))

# 매 프레임 처리 및 화면 출력
while True : 
    ret, frame = cap.read()
    if not ret:
        break

    edge = cv2.Canny(frame, 50, 150)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    out.write(edge)

    cv2.imshow('frame', frame)
    cv2.imshow('edge', edge)

    if cv2.waitKey(10) == 27 :
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
