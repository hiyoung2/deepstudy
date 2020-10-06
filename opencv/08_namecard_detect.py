# 명함 검출 및 인식

# 일반적인 명함 사진의 조건
# - 명함은 흰색 배경에 검정색 글씨이다
# - 명함은 충분히 크게 촬영되었다
# - 명함은 각진 사각형 모양이다

# 명함 검출 및 인식 진행 과정
# 이진화 -> 외곽선 검출&다각형 근사화 -> 투영 변환 -> OCR

import sys
import cv2

src = cv2.imread('./opencv/namecard/namecard1.jpg')

if src is None:
    print("image load failed")
    sys.exit()

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()