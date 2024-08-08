import KalmanBP


def normalize_rescale(x, x_min, x_max):
    normalized = (x - x_min) / (x_max - x_min)
    rescaled = normalized * (0.9 - 0.8) + 0.8
    return rescaled


# 示例
x = 5  # 假设x的原始范围是0到10
x_min = 0
x_max = 10
scaled_x = normalize_rescale(x, x_min, x_max)
print(scaled_x)