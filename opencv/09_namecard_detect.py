# 명함 검출 및 인식

# 일반적인 명함 사진의 조건
# - 명함은 흰색 배경에 검정색 글씨이다
# - 명함은 충분히 크게 촬영되었다
# - 명함은 각진 사각형 모양이다

# 명함 검출 및 인식 진행 과정
# 이진화 -> 외곽선 검출&다각형 근사화 -> 투영 변환 -> OCR

import sys
import cv2

# src = cv2.imread('./opencv/namecard/namecard1.jpg')
# src = cv2.imread('./opencv/namecard/namecard2.jpg') # namecard1 보다 밝게 찍힌 사진
src = cv2.imread('./opencv/namecard/namecard3.jpg') # namecard1 보다 밝게 찍힌 사진

if src is None:
    print("image load failed")
    sys.exit()

# src = cv2.resize(src, (640, 480))
src = cv2.resize(src, (0, 0), fx=0.5, fy=0.5)

# convert color to gray scale
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# binarization 
# _, src_bin = cv2.threshold(src_gray, 130, 255, cv2.THRESH_BINARY)
# using OTSU
th, src_bin = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# OTSU로 자동 임계값을 설정할 것이기 때문에 기존에 줬던 threshold 값 130을 0으로 변경하면 된다
# threshold가 0인 것이 아니라 자동으로 설정되기 때문에 값을 주지 않는 것이라고 보면 된다
# 130을 줘도 무시하고 자동으로 설정되지만 소스코드를 이해하기 더 쉽게끔 이렇게 만들어주는 것이 좋다
print(th) # 151.0, 자동으로 적용된 임계값

cv2.imshow('src', src)
cv2.imshow('src_gray', src_gray)
cv2.imshow('src_bin', src_bin)

cv2.waitKey()
cv2.destroyAllWindows()

# 1. 이진화
# 영상의 이진화(Binarization)란?
# - 영상의 픽셀 값을 0(black) 또는 1(255)(white)로 만드는 연산
# - 배경(Background) vs. 객체(Object)
# - 관심 영역 vs. 비관심 영역

# 그레이스케일 영상의 이진화
# g(x, y) = 0 (if f(x, y) <= T)  /  255 (if f(x, y) > T) -> T : 임계값, threshold

# 임계값 함수
# cv2.threshold(src, thresh, maxval, type, dst=None)->retval, dst
# - src : 입력 영상(다채널, 8비트 또는 32비트 실수형)
# - thresh: 임계값
# - maxval : THRESH_BINARY 또는 THRESH_BINARY_INV 방법을 사용할 때의 최댓값 지정(cv.ThresholdTypes)
# - retval : 사용된 임계값
# - dst : (출력) 임계값 영상(src와 동일 크기, 동일 타입)

# cv2.ThresholdTypes
# cv2.THRESH_BINARY
# cv2.THRESH_BINARY_INV
# 아래의 타입들은 완전한 이진화는 아님
# cv2.THRESH_TRUNC
# cv2.THRESH_TOZERO
# cv2.THRESH_MASK
# 특히 이 아래 2개는 임계값 자동 결정 방법에 속함
# cv2.TRHESH_OTSU : Otsu 알고리즘으로 임계값 결정
# cv2.THRESH_TRIANGLE : 삼각 알고리즘으로 임계값 결정

# 이진화 : 임계값 결정 방법
# 자동 임계값 결정 방법 : Otsu 방법
# - 입력 영상이 배경과 객체 두 개로 구성되어 있다고 가정 -> bimodal histogram
# - 두 픽셀 분포의 분산의 합이 최소가 되는 임계값을 선택(Minimize within-class variance)
# - 효과적인 수식 전개와 재귀식을 이용하여 빠르게 임계값을 결정
# Otus 가 무조건 best 임계값을 설정해주는 건 아니다
# 경우에 따라 Otus에 의해서 정해진 임계값에 우리가 임의로 10을 더하거나 빼줌으로서 오히려 더 좋은 결과가 나올 수도 있다
# 경험치가 중요


