import cv2

img = cv2.imread('./opencv/cat.bmp')

print(type(img)) #<class 'numpy.ndarray'> / 넘파이의 ndarray 다차원 배열형태

print(img.shape) # (480, 640, 3) / 세로, 가로, 컬러(BGR 순서형태로 컬러 정보 저장, 일반적인 순서인 RGB가 아님에 주의!)
print(img.dtype) # uint8