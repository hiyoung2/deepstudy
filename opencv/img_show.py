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

