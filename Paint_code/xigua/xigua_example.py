import cv2
import numpy as np

# 读取图像
image = cv2.imread('xigua_original.png')  # 替换为你的图像路径

# 将图像转换为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用自适应阈值处理灰度图像
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 找到边缘的轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个与输入图像相同大小的空白图像，初始化为黑色
result_image = np.zeros_like(image)

# 在空白图像上绘制最外一层轮廓，黑色部分不变
cv2.drawContours(result_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

# 通过逻辑条件索引将最外一层轮廓内部的白色部分变成绿色
result_image[(result_image == [255, 255, 255]).all(axis=-1) & (image == [255, 255, 255]).all(axis=-1)] = [0, 255, 0]

# 显示结果
cv2.imshow('Result', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('xigua_example.png', result_image)
