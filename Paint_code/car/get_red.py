import cv2
import numpy as np

# 读取图像
image = cv2.imread('car_red.png')

# 将图像从BGR转换为HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置红色的HSV范围
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# 创建红色掩码
red_mask = cv2.inRange(hsv, lower_red, upper_red)

# 寻找红色区域的轮廓
contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 对每个红色点的轮廓进行循环，获取坐标
red_points = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        red_points.append((cx, cy))

# 输出红色点的坐标
print("Red Points Coordinates:", red_points)

# 在原始图像上标记红色点
for point in red_points:
    cv2.circle(image, point, 5, (0, 0, 255), -1)

# 显示结果图像
cv2.imshow('Detected Red Points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
