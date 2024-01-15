import numpy as np

# 四个给定的点
points = np.array([(696, 483), (65, 482), (698, 63), (65, 62)])

# 获取矩形的最小和最大坐标
min_x, min_y = np.min(points, axis=0)
max_x, max_y = np.max(points, axis=0)

# 在矩形内均匀分散100个点
num_points = 100
x_values = np.linspace(min_x, max_x, int(np.sqrt(num_points)))
y_values = np.linspace(min_y, max_y, int(np.sqrt(num_points)))

# 生成均匀分散的点
grid_x, grid_y = np.meshgrid(x_values, y_values)
interpolated_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

# 输出到文件
output_file_path = 'output.txt'
with open(output_file_path, 'w') as file:
    for point in interpolated_points:
        file.write(f"new Vector2({int(point[0])}, {int(point[1])}),\n")

print(f"Unity Vector2 Points written to {output_file_path}")
