import tensorflow as tf
import numpy as np
import cv2
import os
# from PIL import Image, ImageFile


np.random.seed(15)

path = 'D:\\deepstudy\\flow_from_directory'

generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 20, # 회전
    width_shift_range=0.2, # x축 이동
    height_shift_range=0.2, # y축 이동
    rescale = 1. / 255 # 이미지 픽셀 값을 0 ~ 1로 맞춰주기 위해 설정(정규화, MinMaxScaler 효과)
)

batch_size = 4 # 총 4장의 이미지를 한 번에 읽어들이기 위해 4로 잡음
iterations = 5 # 5번의 이미지 증식
images = []

# 생성된 generator의 flow_from_directory를 사용해 이미지 증식을 하는 방식에는 2가지 방식이 있다
# 1. flow_from_directory의 next() 함수 사용
# 2. for문에서 flow_from_directory 사용

# 1번 방법

obj = generator.flow_from_directory(
    path, target_size = (150, 150), batch_size = batch_size, class_mode = 'binary'
)

# flow_from_directory를 obj 라는 이름으로 생성
# 여기서 class_mode는 폴더명에 따라 labeling을 할 때 어떤 방식으로 할 지 정해주는 파라미터이다
# class_mode = 'banary'로 설정시 0 or `로 labelling 한다`

for i in enumerate(range(iterations)):
    img, label = obj.next()
    n_img = len(label)

    base = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR) # keras는 RGB, opencv는 BGR이라 변경함
    for idx in range(n_img-1):
        img2 = cv2.cvtColor(img[idx+1], cv2.COLOR_BGR2RGB)
        base = np.hstack((base, img2))
    images.append(base)

img = images[0]
for idx in range(len(images-1)):
    img = np.vstack((img, images[idx+1]))
cv2.imshow('result', img)

# line34~35 : 핵심부분, line38부터는 증식한 이미지를 한 번에 그리기 위해 추가한 코드
# obj.next() 함수를 한 번 호출할 때마다 obj는 설정된 경로에서 
# batch_size에 맞춰 이미지를 target_size의 크기로, binary 형태로 labelling하며 불러온다
