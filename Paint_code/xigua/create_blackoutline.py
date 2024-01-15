import cv2
import numpy as np

# 加载图像
image = cv2.imread('xigua_gray.png')

# 将图像转换为灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Canny边缘检测找到图像的轮廓
edges = cv2.Canny(gray_image, 30, 100)

# 找到轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在图像外面添加一圈1像素的黑色
result_image = cv2.drawContours(image.copy(), contours, -1, (0, 0, 0), 1)

# 显示原始图像和结果图像
cv2.imshow('Original Image', image)
cv2.imshow('Result Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('xigua_get_outline(black_gray).png',result_image)
