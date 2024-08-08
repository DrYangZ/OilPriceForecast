import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 读取Excel文件
df = pd.read_excel('data1.xlsx')

# 将第一个维度的数据作为因变量y，其他维度的数据作为自变量X
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# 创建岭回归模型
ridge = Ridge(alpha=1.0)  # alpha是正则化强度的参数，可以调整

# 拟合模型
ridge.fit(X, y)

# 输出系数
coefficients = ridge.coef_

# 设置图形大小
plt.figure(figsize=(10, 6))

# 绘制每个自变量及其系数
for i, coef in enumerate(coefficients):
    # 箭头起点和终点
    start = (0.3, 0.8 - i*0.12)
    end = (0.65, 0.8 - len(coefficients)*0.06)
    # 绘制箭头
    plt.annotate('', xy=end, xycoords='axes fraction',
                 xytext=start, textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="->", lw=1.5))
    # 在箭头上方添加系数文本
    plt.text(0.5, 0.85 - i*0.12, f'{coef:.2f}', ha='center', va='center')

# 添加y标签
plt.text(1.0, 0.8 - len(coefficients)*0.06, 'y', ha='right', va='center')

# 添加x标签
for i, col in enumerate(X.columns):
    plt.text(0.4, 0.8 - i*0.12, col, ha='center', va='center')

# 设置图表标题和坐标轴
plt.title('Ridge Regression Coefficients')
plt.xlim(0, 1.5)
plt.ylim(0, 1)
plt.axis('off')  # 不显示坐标轴

# 显示图形
plt.tight_layout()
plt.show()
