import cv2
import numpy as np

# 加载图像
image = cv2.imread('xigua_have_outline.png')

# 将图像转换为灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 设定阈值，将灰色变为黑色
threshold_value = 128
_, binary_mask = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

# 创建一个全黑图像，与原始图像大小相同
black_image = np.zeros_like(image)

# 将二值化掩码应用于原始图像，将灰色变为黑色
black_image[binary_mask > 0] = image[binary_mask > 0]

# 显示原始图像和结果图像
cv2.imshow('Original Image', image)
cv2.imshow('Result Image', black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
