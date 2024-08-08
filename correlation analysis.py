import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import mpl

# 显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 显示负号
mpl.rcParams['axes.unicode_minus'] = False

# 读入数据
data = pd.read_excel('data1.xlsx', index_col=0)

# 相关性分析
pearson_corr = data.corr(method='pearson')
spearman_corr = data.corr(method='spearman')
kendall_corr = data.corr(method='kendall')

# 设置图形大小
plt.figure(figsize=(10, 8))

# 绘制并保存皮尔逊相关性热力图
plt.subplot(1, 1, 1)
sns.heatmap(pearson_corr, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.1f', cmap='RdBu_r')
plt.title('pearson')
plt.savefig('pearson_result.png')
plt.clf()  # 清除当前图形

# 绘制并保存Spearman相关性热力图
plt.subplot(1, 1, 1)
sns.heatmap(spearman_corr, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.1f', cmap='RdBu_r')
plt.title('Spearman')
plt.savefig('spearman_result.png')
plt.clf()  # 清除当前图形

# 绘制并保存Kendall相关性热力图
plt.subplot(1, 1, 1)
sns.heatmap(kendall_corr, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True, fmt='.1f', cmap='RdBu_r')
plt.title('Kendall')
plt.savefig('kendall_result.png')
plt.clf()  # 清除当前图形

# 显示所有图形
plt.show()
