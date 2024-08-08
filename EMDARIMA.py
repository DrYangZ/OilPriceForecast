import math
import os
import zipfile
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from PyEMD import EMD
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import openpyxl
from pylab import mpl

plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.sans-serif'] = ['SimHei']





def main(data, price, date_time, waterlevel1, waterlevel2):
    dates = data
    prices = price

    # 创建DataFrame
    Data = pd.DataFrame({'Date': pd.to_datetime(dates), 'Price': prices})
    Data.set_index('Date', inplace=True)


    # EMD分解
    emd = EMD()
    imfs = emd.emd(Data['Price'].values)
    n_imfs = imfs.shape[0]


    def plot_arima_predictions(Data, order, seasonal_order=None):
        model = SARIMAX(Data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        predictions = model_fit.forecast(steps=12)

        print(model_fit.summary())

    # 对每个IMF分量应用ARIMA模型
    # IMF1 - ARIMA(2,0,10)x(2,0,0,12)
    plot_arima_predictions(imfs[0], order=(2, 0, 10), seasonal_order=(2, 0, 0, 12))

    # IMF2 - ARIMA(4,0,7)
    if n_imfs > 1:
        plot_arima_predictions(imfs[1], order=(4, 0, 7))

    # IMF3 - ARIMA(6,1,0)
    if n_imfs > 2:
        plot_arima_predictions(imfs[2], order=(6, 1, 0))

    # 预测函数
    def predict_with_arima(Data, order, steps=36):
        model = ARIMA(Data, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast

    # 总预测结果
    total_forecast = np.zeros(36)

    # 对每个IMF分量应用ARIMA模型并累加预测结果
    for i in range(n_imfs):
        # 此处假设为每个IMF分量选择了合适的ARIMA参数
        # 这些参数可能需要根据实际数据进行调整
        order = (5, 1, 0)  # 示例参数，需要根据实际情况调整
        forecast = predict_with_arima(imfs[i], order)
        total_forecast += forecast

    # 绘制预测结果
    future_date = pd.date_range(start='2024-01', periods=36, freq='ME')
    plt.figure(figsize=(12, 6))
    plt.plot(Data.index, Data['Price'], label='Historical Data')
    plt.plot(future_date, total_forecast, label='Forecast', linestyle='--')
    plt.title('长江成品油运价预测')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    # plt.savefig('./predict_fig/' + oil_type + '_' + start_port + '_' + end_port + '_' + load_level + '.png')
    # plt.show()
    # 总预测结果
    total_forecast = np.zeros(36)

    # 对每个IMF分量应用ARIMA模型并累加预测结果
    for i in range(n_imfs):
        # 此处假设为每个IMF分量选择了合适的ARIMA参数
        # 这些参数可能需要根据实际数据进行调整
        order = (5, 1, 0)  # 示例参数，需要根据实际情况调整
        forecast = predict_with_arima(imfs[i], order)
        total_forecast += forecast

    # 生成未来的时间列表
    start_date = datetime.strptime(date_time, '%Y-%m')
    date_list = []
    for i in range(12):
        new_date = start_date.replace(year=start_date.year + (start_date.month + i -1) // 12,
                                      month=(start_date.month + i - 1) % 12 + 1)
        date_list.append(new_date.strftime('%Y-%m'))

    future_date = date_list


    # price_predictout = [x * (((x - x_min) / (x_max - x_min)) * (1.2 - 1.0) + 1.0) for x in
    #                     total_forecast[:12]]
    # (math.atan(x)/1.57)*0.4 + 0.8
    price_predictout = [i*((math.atan(waterlevel2*0.3 + waterlevel1*0.7)/1.57)*0.4 + 0.8) for i in
                        total_forecast[:12]]

    # 绘制预测结果
    # future_date = pd.date_range(start='2024-01', periods=36, freq='ME')
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, label='Historical Data')
    plt.plot(future_date[0], price_predictout[0], marker='*', label='Forecast', linestyle='--')
    plt.xticks(rotation=45)
    plt.title('长江成品油运价预测')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.savefig("output" + "\\" + "nextyear_pre.png")
    # plt.show()

    # future_dates = pd.date_range(start='2024-01-01', periods=36, freq='ME')

    # 保存预测的值
    data_series = pd.Series([price_predictout[0]], index=[future_date[0]], name='Data')
    data_series.index = data_series.index.astype(str)
    excel_file = 'output//nextyear_pre.xlsx'
    data_series.to_excel(excel_file)

    with open("out_pre.txt", "w+") as f:
        content3 = str(data_series)
        f.write(content3)

    get_files_path = "output"  # 需要压缩的文件夹
    set_files_path = "output.zip"  # 存放的压缩文件地址(注意:不能与上述压缩文件夹一样)
    f = zipfile.ZipFile(set_files_path, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(get_files_path):
        fpath = dirpath.replace(get_files_path, '')  # 注意2
        fpath = fpath and fpath + os.sep or ''  # 注意2
        for filename in filenames:
            f.write(os.path.join(dirpath, filename), fpath + filename)
    f.close()
