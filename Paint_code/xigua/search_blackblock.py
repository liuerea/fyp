import cv2
import numpy as np

# 加载图像
image = cv2.imread('xigua_get_outline(black_gray).png')

# 将图像转换为灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 找到所有黑色像素的坐标
black_pixel_coordinates = np.column_stack(np.where(gray_image == 0))

# 隔20个点取一个坐标
selected_coordinates = black_pixel_coordinates[::20]

# 将坐标转换为 y * 960 + x 格式
converted_coordinates = [coord[0] * gray_image.shape[1] + coord[1] for coord in selected_coordinates]

# 将坐标输出为C#中可以定义的数组格式
output_str = "int[] blackPixelCoordinates = new int[] {\n"
output_str += ", ".join(map(str, converted_coordinates))
output_str += "};"

# 将输出字符串写入C#文件
output_file_path = 'BlackPixelCoordinates.cs'
with open(output_file_path, 'w') as file:
    file.write(output_str)

print(f"Converted coordinates have been saved to {output_file_path}")
