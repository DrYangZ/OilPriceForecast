# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
import sys
from collections import deque
import warnings
import io

# 设置标准输出的编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
def read_data(file_path, sheet_name):
    # 读取Excel文件
    file_path = file_path
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df['时间'] = pd.to_datetime(df['时间'], format='%Y.%m')
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


# 定义函数来训练 ARIMA 模型并进行预测
def arima_forecast(series, steps):
    model = ARIMA(series, order=(2, 1, 2))  # 使用 ARIMA(2, 1, 2)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast


# 定义函数来训练随机森林模型并进行预测
def rf_forecast(series, steps):
    # 创建滞后特征
    X = []
    y = []
    for i in range(5, len(series)):
        X.append(series[i - 5:i])  # 使用前5个数据点作为特征
        y.append(series[i])  # 当前点作为目标值

    X = np.array(X)
    y = np.array(y)

    # print(f"X序列为:\n{X},\ny序列为:\n{y}")
    # 训练随机森林回归模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 用最后5个数据点作为特征进行预测
    forecast = []
    current_input = series[-5:]

    for _ in range(steps):
        prediction = model.predict([current_input])[0]
        forecast.append(prediction)
        current_input = np.roll(current_input, -1)  # 移动窗口
        current_input[-1] = prediction

    return forecast


# 定义组合模型进行预测
def arima_rf_forecast(series, steps):
    # 使用 ARIMA 模型预测
    arima_result = arima_forecast(series, steps)

    # 使用随机森林预测
    rf_result = rf_forecast(series, steps)

    # 将结果转换为 NumPy 数组
    arima_result = np.array(arima_result)
    rf_result = np.array(rf_result)

    # 加权组合结果 (ARIMA 0.88, RF 0.12)
    combined_result = 0.88 * arima_result + 0.12 * rf_result

    # print(combined_result)
    return combined_result


def main(file_path='./Datasets.xlsx', sheet_name='南京到重庆-贸易商', forecast_steps=6):
    # 设置 matplotlib 使用的字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore', category=UserWarning)
    df = read_data(file_path=file_path, sheet_name=sheet_name)
    # print(df.index[-1])
    forecast_value_dict = {}

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
        # print(df[label].values)
        mean_value = df[label].mean()
        # print(mean_value)
        labels[label].append(mean_value)
        # print(df[label].shape)
        # print(df[label].ndim)
        forecast_value_dict[label] = arima_rf_forecast(df[label].values, steps=forecast_steps)

    # print(forecast_value_dict)
    # print(labels)
    forecast_price = arima_rf_forecast(df['价格'].values, steps=forecast_steps)
    target_price = deque()
    for n in range(forecast_steps):
        influence = 0
        for i, value in enumerate(list(zip(*forecast_value_dict.values()))[n]):
            influence += (value - list(zip(*labels.values()))[1][i]) * list(zip(*labels.values()))[0][i]
        target_price.append(forecast_price[n] + influence)
    # print(target_price, '\n', len(target_price))
    # print(df['价格'].iloc[-1])
    target_price.appendleft(df['价格'].iloc[-1])
    # print(target_price)

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
    plt.title('RF-ARIMA模型预测结果', fontsize=16, fontweight='bold')
    plt.xlabel('时间', fontsize=14)
    plt.ylabel('价格', fontsize=14)
    # 添加网格线
    plt.grid(True, which="both", ls="--", c='0.65')
    # 添加图例
    plt.legend(fontsize=12, framealpha=0.5)
    # 调整坐标轴的刻度
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # 调整布局并显示图形
    plt.tight_layout()

    plt.savefig('./fig/RF-ARIMA.png', dpi=300)
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
    save_path = './predict_data/RF-ARIMA.csv'
    display_data.to_csv(save_path, index=False)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1:]
        print(f'EMD-ARIMA.py script has received arguments values from sys.argv{arg}')
        main(arg[0], arg[1], int(arg[2]))
    # main()