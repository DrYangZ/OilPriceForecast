import pandas as pd
import matplotlib.pyplot as plt
import PyEMD as emd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


def draw_series(df):
    column_num = 4
    row_num = len(df.columns[4:]) // column_num
    # print(row_num)
    fig, axs = plt.subplots(row_num, column_num, figsize=(20, 10))
    plt.style.use('bmh')
    if row_num == 1:
        axs = [axs]
    for r in range(row_num):
        for c in range(column_num):
            if r * row_num + c < len(df.columns[4:]):
                col_name = df.columns[4 + r * row_num + c]
                axs[r, c].plot(df.index, df[col_name], color='#1F4E79', label=f'{col_name}序列', lw=2)
                axs[r, c].set_title(f'2021年1月-2023年10月{col_name}序列', fontsize=8)
                axs[r, c].set_xlabel('时间', fontsize=8)
                axs[r, c].set_ylabel(col_name, fontsize=8)
                axs[r, c].grid(True, which='both', ls='--', c='0.65', linewidth=0.5)
                # 添加图例
                # axs[r, c].legend(fontsize=8, framealpha=0.5)
                # 调整坐标轴的刻度
                axs[r, c].tick_params(axis='x', labelsize=8)
                axs[r, c].tick_params(axis='y', labelsize=8)
            else:
                # 如果没有足够的子图，隐藏多余的 Axes
                axs[r, c].axis('off')
    # 调整布局并显示图形
    plt.tight_layout()
    plt.style.use('bmh')
    plt.show()
    plt.savefig('./fig/articles/all_series.png')
    plt.close()


def stability_test(series):
    result = adfuller(series.dropna(), autolag='AIC')  # dropna() to handle any missing values
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    # 可选：打印更多的ADF检验结果
    print(f'Used Lag: {result[2]}')
    print(f'Number of Observations: {result[3]}')
    for key, value in result[4].items():
        print(f'Critical Value ({key}): {value}')
    # 判断并打印时间序列是否平稳
    if result[1] <= 0.05:
        print("在5%显著性水平下，拒绝原假设，时间序列是平稳的。")
    else:
        print("在5%显著性水平下，不能拒绝原假设，时间序列可能是非平稳的。")


def draw_imfs(series):
    # 提取 '价格' 列作为信号
    signal = series.values
    # 使用 PyEMD 库进行 EMD 分解
    e = emd.EMD()
    imfs = e.emd(signal, None)
    # 创建绘图
    plt.figure(figsize=(12, 6))
    # 绘制原始信号
    plt.subplot(len(imfs) + 1, 1, 1)
    plt.plot(df.index, signal, label='Original Signal', color='#1F4E79', lw=2)
    plt.xlabel("时间")
    plt.ylabel("幅度")
    plt.title('原始信号')
    plt.grid(True, which='both', ls='--', c='0.65', linewidth=0.5)
    plt.legend(loc="upper right")
    # 绘制每个 IMF
    for i, imf in enumerate(imfs, start=1):
        plt.subplot(len(imfs) + 1, 1, i + 1)
        plt.plot(df.index, imf, label=f'IMF {i}', color='#1F4E79', lw=2)
        plt.xlabel("时间")
        plt.ylabel("幅度")
        plt.grid(True, which='both', ls='--', c='0.65', linewidth=0.5)
        plt.title(f'IMF {i}')
        plt.legend(loc="upper right")
    plt.style.use('bmh')
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    plt.show()
    plt.close()


def draw_diff(series):
    # 计算一阶差分
    df_diff = series.diff()

    # 绘制一阶差分图
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df_diff, label='序列一阶差分', color='#1F4E79', lw=2)
    plt.title('序列一阶差分')
    plt.xlabel('Index')
    plt.ylabel('Difference')
    plt.legend()
    plt.style.use('bmh')
    plt.grid(True, which='both', ls='--', c='0.65', linewidth=0.5)
    plt.show()


def draw_acf_pacf(series):
    # 绘制 ACF 图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sm.graphics.tsa.plot_acf(series, lags=17, ax=ax1)
    ax1.set_title('Autocorrelation')
    ax1.grid(True, which='both', ls='--', c='0.65', linewidth=0.5)
    # 绘制 PACF 图
    sm.graphics.tsa.plot_pacf(series, lags=17, ax=ax2)
    ax2.set_title('Partial Autocorrelation')
    ax2.grid(True, which='both', ls='--', c='0.65', linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    excel_path = './Datasets.xlsx'
    df = pd.read_excel(excel_path)
    df.set_index(df['时间'], inplace=True)

    label = '价格'
    series = df[label]

    # 以子图的方式将dataframe所有序列绘制在一张图中
    # draw_series(df)

    # 针对某序列进行数据平稳性检验
    # stability_test(series)

    # 绘制IMFS分量图
    # draw_imfs(series)

    # 绘制一阶差分图
    # draw_diff(series)

    # 绘制acf和pacf图
    # draw_acf_pacf(series)

    # 打印印象因素列表
    print(df.columns[4:])