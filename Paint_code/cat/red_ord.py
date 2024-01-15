import cv2
import numpy as np

# 读取图像
image = cv2.imread('red_cat.png')

# 查找图像中 RGB 值为 (255, 0, 0) 的像素点坐标
red_pixel_coordinates = np.column_stack(np.where(np.all(image == [0, 0, 255], axis=-1)))

# 输出到Unity Vector2格式的文件
output_file_path = 'red_pixel_coordinates.txt'
with open(output_file_path, 'w') as file:
    for point in red_pixel_coordinates:
        file.write(f"new Vector2({point[1]}, {point[0]}),\n")

print(f"Red Pixel Coordinates with RGB (255, 0, 0) written to {output_file_path}")
