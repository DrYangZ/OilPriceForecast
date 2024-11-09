# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EMD
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
from collections import deque
import sys
import io

# 设置标准输出的编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
def read_data(file_path, sheet_name):
    # 读取Excel文件
    file_path = file_path
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df['时间'] = pd.to_datetime(df['时间'], format='%Y.%m')
    df['月份'] = df['时间'].dt.strftime('%m').astype(int)
    df.set_index('时间', inplace=True)
    df.index.freq = 'MS'  # 设置频率信息为每月开始

    '''
    # 数据可视化
    price_series = df['价格'].values
    distance = df['航线距离']
    price_index = df['成品油运价指数']
    petrol_price = df['汽油价格（元/吨）']
    diesel_price = df['柴油价格（元/吨）月底价格']
    water_level_status = df['丰水期和枯水期']
    water_level_data = df['水位（m）']
    water_transport_employee_number = df['水上运输业从业人员数']
    tube_transport_employee_number = df['管道运输业从业人员数']
    GDP_index = df['GDP指数']
    CPI = df['CPI']
    employment_rate = df['就业率']
    consumer_price_index = df['国家财政收入（居民消费价格指数）']
    energy_consumption = df['能源消费总量']
    petrol_consumption = df['石油能源消费总量(万吨)	']
    port_throughput = df['内河港口石油天然气进出港吞吐量']
    refined_oil_consumption = df['成品油消费量']
    transport_capacity = df['运力保有量']
    tube_scale = df['管道运输规模']
    dam_status = df['待坝情况']
    time_series = df.index
    # print(water_level_data)
    '''
    return df


def draw_data(label, df):
    # plt.ion()  # 启用交互模式
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[label], label=label)
    plt.title(f'{label}时间序列')
    plt.xlabel('时间')
    plt.ylabel(f'{label}')
    plt.legend()
    plt.show()


def emd_draw(label, df):
    if df[label].nunique() == 1:
        print('------------------------------------')
        print(f'"{label}"为常值序列，无法进行arima预测。')
        print('------------------------------------')
        return
    emd = EMD()
    imfs = emd.emd(df[label])

    # 获取所有IMF分量和residual
    imf_components = imfs[:-1]
    residual = imfs[-1]

    # 创建一个新的图像
    fig, axs = plt.subplots(len(imfs), 1, figsize=(10, 4 * len(imfs)))

    # 如果只有一个子图，axs将会是一个单个的Axes对象而不是列表
    if not isinstance(axs, np.ndarray):
        axs = [axs]

    # 绘制每一个IMF分量
    for i, imf in enumerate(imf_components):
        axs[i].plot(df.index, imf, label=f'{label}_IMF {i + 1}')
        axs[i].set_title(f'{label}_IMF {i + 1}')
        axs[i].legend()

    # 绘制residual
    axs[-1].plot(df.index, residual, label=f'{label}_Residual')
    axs[-1].set_title(f'{label}_Residual')
    axs[-1].legend()
    plt.tight_layout()  # 自动调整子图布局
    plt.show()

    for i, imf in enumerate(imfs):
        data = adfuller(imf)
        adf_statistic = data[0]
        p_value = data[1]
        # 判断是否平稳
        if p_value <= 0.05:
            print(f"序列{label}_imf{i + 1}是平稳的")
        else:
            print(f"序列{label}_imf{i + 1}不是平稳的")


def arima_draw(label, df, steps=12):

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    series = df[label]
    # 检查序列是否为常量
    if series.nunique() == 1:
        print('------------------------------------')
        print(f'"{label}"为常值序列，无法进行arima预测。')
        print('------------------------------------')
        return [df[label].mean()] * steps
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] > 0.05:
        print('Time series is not stationary.')
    else:
        print('Time series is stationary.')

    # 训练 ARIMA 模型
    model = ARIMA(series, order=(2, 1, 2))
    model_fit = model.fit()

    # 预测
    forecast = model_fit.forecast(steps=steps)
    forecast_value = forecast.values
    # print(f'{label}预测结果为：\n{forecast.values}\n{type(forecast.values)}')
    # print(forecast_value[1], type(forecast_value[1]))

    # # 绘制结果
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(len(df)), series, label=f'原始{label}序列')
    # if steps > 1:
    #     plt.plot(np.arange(len(df), len(df)+steps), forecast, color='red', label=f'{label}的预测序列')
    # elif steps == 1:
    #     plt.plot(len(df)+steps, forecast, color='red',marker='o', label=f'{label}的预测点')
    # plt.title(f'{label}的预测结果')
    # plt.xlabel('时间')
    # plt.ylabel(f'{label}')
    # plt.legend()
    # plt.show()

    return forecast_value


def main(file_path='./Datasets.xlsx', sheet_name='南京到重庆-贸易商', forecast_steps=6):
    warnings.filterwarnings("ignore")
    df = read_data(file_path=file_path, sheet_name=sheet_name)
    forecast_value_dict = {}
    predict_img_path = './fig/EMD-ARIMA.png'

    labels = {
        '航线距离': [0.015],
        '成品油运价指数': [0.0002],
        '汽油价格（元/吨）': [0.0012],
        '柴油价格（元/吨）月底价格': [0.0012],
        '丰水期和枯水期': [0.007],
        '水位（m）': [0.006],
        '水上运输业从业人员数': [0.0003],
        '管道运输业从业人员数': [0.0002],
        'GDP指数': [0.0006],
        'CPI': [0.0005],
        '就业率': [0.0003],
        '国家财政收入（居民消费价格指数）': [0.0003],
        '能源消费总量': [0.0004],
        '石油能源消费总量(万吨)': [0.0005],
        '内河港口石油天然气进出港吞吐量': [0.0008],
        '成品油消费量': [0.0007],
        '运力保有量': [0.0005],
        '管道运输规模': [0.0003],
        '待坝情况': [0.002],
    }

    for i, label in enumerate(labels):
        mean_value = df[label].mean()
        # print(f'{label}历史时序数据的基准值为：\n{mean_value}')
        labels[label].append(mean_value)
        # draw_data(label=label, df=df)
        # emd_draw(label=label, df=df)
        # 确定 ARIMA 模型的参数
        forecast_value_dict[label] = arima_draw(label=label, df=df, steps=forecast_steps)
    forecast_price = arima_draw(label='价格', df=df, steps=forecast_steps)
    # print(list(zip(*forecast_value_dict.values())))
    # print(forecast_price)
    # print(list(zip(*labels.values())))
    # print(len(list(zip(*labels.values()))[0]), len(list(zip(*labels.values()))[1]))
    target_price = deque()
    for n in range(forecast_steps):
        influence = 0
        for i, value in enumerate(list(zip(*forecast_value_dict.values()))[n]):
            influence += (value-list(zip(*labels.values()))[1][i]) * list(zip(*labels.values()))[0][i]
        target_price.append(forecast_price[n]+influence)
    # print(target_price, '\n', len(target_price))
    # print(df['价格'].iloc[-1])
    target_price.appendleft(df['价格'].iloc[-1])
    # print(target_price)

    # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(len(df)), df['价格'], label='原始价格序列')
    # if forecast_steps > 1:
    #     plt.plot(np.arange(len(df)-1, len(df)+forecast_steps), target_price, color='red', label=f'价格的最终预测序列')
    # elif forecast_steps == 1:
    #     plt.plot(len(df)+forecast_steps, target_price, color='red',marker='o', label=f'价格的最终预测点')
    # plt.title(f'最终预测价格的结果')
    # plt.xlabel('时间')
    # plt.ylabel(f'最终预测价格')
    # plt.legend()

    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')  # 使用ggplot风格
    # 绘制原始价格序列
    plt.plot(np.arange(len(df)), df['价格'], label='原始价格序列', lw=2, color='#1F4E79')
    if forecast_steps > 1:
        # 绘制预测序列
        plt.plot(np.arange(len(df) - 1, len(df) + forecast_steps), target_price,
                 color='red', label=f'价格的最终预测序列', linestyle='--', lw=2)
    elif forecast_steps == 1:
        # 标记预测点
        plt.scatter([len(df) + forecast_steps], [target_price], color='red', marker='o', s=100,
                    label=f'价格的最终预测点')
    # 设置标题和标签
    plt.title('EMD-ARIMA模型预测结果', fontsize=16, fontweight='bold')
    plt.xlabel('时间', fontsize=14)
    plt.ylabel('价格', fontsize=14)
    # 添加网格线
    plt.grid(True, which="both", ls="--", c='0.65')
    # 添加图例
    plt.legend(fontsize=12, framealpha=0.5)
    # 调整坐标轴的刻度
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.savefig(predict_img_path, dpi=300)
    # plt.show()
    plt.close()
    result = list(target_price)[1:]
    predict_date = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_steps,
                                 freq='ME').strftime('%Y-%m').values
    # print(predict_date.strftime('%Y-%m').values)
    # print(result)
    display_data = pd.DataFrame({
        '时间': predict_date,
        '价格': result
    })
    print(display_data)
    save_path = './predict_data/EMD-ARIMA.csv'
    display_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1:]
        print(f'EMD-ARIMA.py script has received arguments values from sys.argv{arg}')
        main(arg[0], arg[1], int(arg[2]))
    # main()