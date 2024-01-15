import cv2
import numpy as np

# 加载图像
image = cv2.imread('xigua_original.png')

# 创建一个掩码，选择黑色区域
black_mask = np.all(image == [0, 0, 0], axis=-1)

# 将黑色部分替换为灰色
image[black_mask] = [128, 128, 128]  # 这里的 [128, 128, 128] 是灰色的BGR值

# 显示原始图像和结果图像
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('xigua_gray.png',image)