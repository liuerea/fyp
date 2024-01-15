import cv2
import numpy as np

# 读取图像
image = cv2.imread('xigua_gray.png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行边缘检测
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 寻找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个与输入图像相同大小的白色图像
result = np.ones_like(image) * 255

# 在新图像上绘制轮廓
cv2.drawContours(result, contours, -1, (0, 0, 0), 1)  # 添加黑色边框

# 显示结果图像
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('xigua_get_outline.png',result)