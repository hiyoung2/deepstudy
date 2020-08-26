import tensorflow as tf
import numpy as np
import cv2
 
np.random.seed(15)
 
path = 'data'
 
generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rescale = 1. / 255)
 
batch_size = 4
iterations = 5
images = []


obj = generator.flow_from_directory(
    path,
    target_size = (150, 150),
    batch_size = batch_size,
    class_mode = 'binary')
 
 
for i in enumerate(range(iterations)):
    img, label = obj .next()
    n_img = len(label)
    
    base = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)  # keras는 RGB, openCV는 BGR이라 변경함
    for idx in range(n_img - 1):
        img2 = cv2.cvtColor(img[idx + 1], cv2.COLOR_RGB2BGR)
        base = np.hstack((base, img2))
    images.append(base)
 
img = images[0]
for idx in range(len(images) - 1):
    img = np.vstack((img, images[idx + 1]))


cv2.imshow('result', img)
cv2.waitKey()
# cv2.waitKey()를 꼭 함께 써줘야한다
# 이 줄을 안쓴다면 다음과 같이 창은 뜨지만 사진은 나오지 않는 오류가 뜬다