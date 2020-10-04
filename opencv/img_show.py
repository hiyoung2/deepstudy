import cv2
import sys

img = cv2.imread('./opencv/cat.bmp')
# img = cv2.imread('./opencv/cat_.bmp') # Image load failed!

# image를 읽어오지 못할 경우의 에러 대비
if img is None:
    print("Image load failed!")
    sys.exit()

print(type(img)) #<class 'numpy.ndarray'> / 넘파이의 ndarray 다차원 배열형태
print(img.shape) # (480, 640, 3) / 세로, 가로, 컬러(BGR 순서형태로 컬러 정보 저장, 일반적인 순서인 RGB가 아님에 주의!)
print(img.dtype) # uint8

# namedWindow : 새로운 창을 만들어주는 함수, 창의 이름 == image
cv2.namedWindow('image')
cv2.imshow('image', img) # image라는 창에다가 img 영상(이미지)을 출력
cv2.waitKey() 
# key 입력을 기다리는 함수, 인자에 숫자를 주면 예를 들어 2000 -> 2000m/s 즉, 2초를 기다리고 프로그램이 종료된다
# 아무런 숫자를 안 쓰거나 0을 쓰면 키보드 입력을 무한대로 기다리게 된다
# img 출력 후 창을 닫고 싶다면 키보드의 어떤 키든 누르면 종료된다
# cv2.destroyAllWindows() # 만들어져 있는 모든 창을 닫는 함수, 생략해도 상관 없음


'''
import cv2

img = cv.imread('./opencv/cat.bmp')
cv2.namedWindow('image')
cv2.imshow('image', img)
cv2.waitKey()

opencv에서는 이 5줄의 코드로 어떤 영상을 불러 와서 화면에 출력하는 것이 가능
사실 cv2.namedWindow('image') 이 코드가 없어도 된다
imshow 실행하면 'image'라는 창이 없어도 알아서 생성하기 때문!
'''

# OpenCV API

# 1. 영상 파일 불러오기
# cv2.imread(filename, flags=None) -> retval
# 1) filename : 불러올 영상 파일 이름(문자열)
# (1) 상대 경로 : 'cat.bmp', '../data/cat.bmp'
# (2) 절대 경로 : 'c:\cat.bmp', '/home/id/cat.bmp'
# 2) flags : 영상 파일 불러오기 옵션 플래그
# (1) cv2.IMREAD_COLOR : BGR 컬러 영상으로 읽기(default), shape = (rows, cols, 3)
# (2) cv2.IMREAD_GRAYSCALE : 그레이스케일 영상으로 읽기, shape = (rows, cols)
# (3) cv2.IMREAD_UNCHANGED : 영상 파일 속성 그대로 읽기(e.g. 투명한 PNG파일 : shape = (rows, cols, 4))
# retval : 불러온 영상 데이터(numpy.ndarray)

# opencv 관련 함수들은 document로 보면 된다 https://docs.opencv.org/master/

# 2. 영상 파일 저장하기
# cv2.imwrite(filename, img, params = None) -> retval
# filename : 저장할 영상 파일 이름(문자열)
# img : 저장할 영상 데이터(numpy.ndarray)
# params : 파일 저장 옵션 지정(속성&값의 정수 쌍)
# e.g. JPG 파일 압축률을 90%로 지정하고 싶다면? [cv2.IMWRITE_JPEG_QUALITY,90] 지정
# retval : 정상적으로 저장하면 True, 실패하면 False

# 3. 새창 띄우기 & 창 닫기
# cv2.namedWindow(winname, flags=None) -> None
# winname : 창 고유 이름, 이 이름으로 창을 구분함
# flags : 창 속성 지정 플래그
# 1) cv2.WINDOW_NORMAL : 영상 크기가 창 크기에 맞게 지정됨
# 2) cv2.WINDOW_AUTOSIZE : 창 크기가 영상 크기에 맞게 자동으로 변경됨(default)

# cv2.destroyWindow(winname) -> None
# cv2.destroyAllWindows() -> None
# winname : 닫고자 하는 창 이름
# 일반적인 경우 프로그램 종료 시 운영 체제에 의해 열려 있는 모든 창이 자동으로 닫힘

# 4. 창 위치 & 크기 지정
# cv2.moveWinidow(winname, x, y) -> None
# winname : 창 이름
# x, y : 이동할 위치 좌표

# cv2.resizeWindow(winname, width, height) -> None
# winname : 창 이름
# width, height : 변경할 창 크기
# 참고 사항
# * 창 생성시 cv2.WINDOW_NORMAL 속성으로 생성되어야 동작함
# * 영상 출력 부분의 크기만을 고려함(제목 표시줄, 창 경계는 고려되지 않음)
