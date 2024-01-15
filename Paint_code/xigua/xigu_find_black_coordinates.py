import cv2
import numpy as np

# 加载图像
image = cv2.imread('xigua_get_outline(black_gray).png')

# 检查图像是否加载成功
if image is None:
    print("Error: Unable to load the image.")
    exit()

# 将图像转换为灰度
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 找到所有黑色像素的坐标
black_pixel_coordinates = np.column_stack(np.where(gray_image == 0))

# 隔40个点取一个坐标
selected_coordinates = black_pixel_coordinates[::400]

# 将坐标输出为Vector2的数据格式
output_str = "Vector2[] blackPixelCoordinates = new Vector2[] {\n"
for coord in selected_coordinates:
    output_str += f"    new Vector2({coord[1]}, {coord[0]}),\n"

output_str += "};"

# 将输出字符串写入C#文件
output_file_path = 'BlackPixelCoordinates.cs'
with open(output_file_path, 'w') as file:
    file.write(output_str)

print(f"Black pixel coordinates have been saved to {output_file_path}")
