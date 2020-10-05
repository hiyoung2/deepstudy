import matplotlib.pyplot as plt
import cv2

# (1) print color-image(video)
imgBGR = cv2.imread('./opencv/cat.bmp')
imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(imgRGB)
# plt.show()

# (2) print grayscale-image(video)
imgGray = cv2.imread('./opencv/cat.bmp', cv2.IMREAD_GRAYSCALE)
plt.axis('off')
plt.imshow(imgGray, cmap='gray')
# plt.show()

# (3) print both (1) and (2)
plt.subplot(121), plt.axis('off'), plt.imshow(imgRGB)
plt.subplot(122), plt.axis('off'), plt.imshow(imgGray, cmap='gray')
plt.show()

