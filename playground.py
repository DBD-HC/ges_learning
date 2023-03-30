import numpy as np
from scipy.ndimage import gaussian_filter

# 创建一个 15x15 的高斯核数组
sigma = 6  # 标准差
window_size = 15
half = window_size // 2
x = np.array([i if i <= half else window_size-i-1 for i in range(window_size)])
y = np.array([i if i <= half else window_size-i-1 for i in range(window_size)])
x, y = np.meshgrid(x, y)
gaussian_kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

# 对高斯核进行归一化
gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)

print(gaussian_kernel)
