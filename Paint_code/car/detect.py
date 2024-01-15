import cv2
import numpy as np

# 读取图像并调整大小
image = cv2.imread('car_red.png')
if image is not None:
    image = cv2.resize(image, (960, 800))

    # 查找图像中 RGB 值为 (255, 0, 0) 的所有像素点坐标
    red_pixel_coordinates = np.column_stack(np.where(np.all(image == [0, 0, 255], axis=-1)))

    # 每400个点取一个
    selected_red_pixel_coordinates = red_pixel_coordinates[::1000]

    # 输出到Unity Vector2格式的文件
    output_file_path = 'selected_red_pixel_coordinates.txt'
    with open(output_file_path, 'w') as file:
        for point in selected_red_pixel_coordinates:
            file.write(f"new Vector2({point[1]}, {point[0]}),\n")

    print(f"Selected Red Pixel Coordinates with RGB (255, 0, 0) written to {output_file_path}")
else:
    print("Error: Could not open or read the image.")
