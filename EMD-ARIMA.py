import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from PyEMD import EMD
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import openpyxl

plt.rcParams['font.sans-serif'] = ['SimHei']

path = './price_data.xlsx'
wb = openpyxl.load_workbook(path)
sheet = wb['Sheet1']

date = [cell.value for cell in sheet['E1:AN1'][0]]
print(date)

# data_row = 2
# data = [cell.value for cell in sheet['E'+str(data_row)+':AN'+str(data_row)][0]]
# print(data)

for row in range(2, 28, 1):
    data_row = row
    data = [float(cell.value) for cell in sheet['E' + str(data_row) + ':AN' + str(data_row)][0]]
    oil_type = sheet['A' + str(row)].value
    start_port = sheet['B' + str(row)].value
    end_port = sheet['C' + str(row)].value
    load_level = sheet['D' + str(row)].value
    print(data)
    print(row,oil_type, start_port, end_port, load_level)

    # 创建DataFrame
    Data = pd.DataFrame({'Date': pd.to_datetime(date, format='%Y-%m'), 'Price': data})
    Data.set_index('Date', inplace=True)

    # 绘制时序图
    plt.figure(figsize=(12, 6))
    plt.plot(Data.index, Data['Price'], marker='o', linestyle='-')
    plt.title('长江成品油运价时序图')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.grid(True)
    plt.show()

    # 进行ADF非平稳性检验
    try:
        adf_test = adfuller(Data['Price'])
    except ValueError:
        print("数据读取结束！")
        break
    print("ADF统计量: ", adf_test[0])
    print("p值: ", adf_test[1])
    print("使用的滞后数: ", adf_test[2])
    print("使用的观测数: ", adf_test[3])
    print("临界值: ", adf_test[4])
    if adf_test[1] < 0.05:
        print("数据是平稳的")
    else:
        print("数据是非平稳的")

    # EMD分解
    emd = EMD()
    imfs = emd.emd(Data['Price'].values)
    n_imfs = imfs.shape[0]

    # 绘制原始数据和EMD分解结果
    plt.figure(figsize=(12, 2 * (n_imfs + 1)))
    plt.subplot(n_imfs + 1, 1, 1)
    plt.plot(Data.index, Data['Price'], label='Original Data')
    plt.title('原始时序数据')
    plt.legend()

    for n in range(n_imfs):
        plt.subplot(n_imfs + 1, 1, n + 2)
        plt.plot(Data.index, imfs[n], label=f'IMF {n + 1}')
        print(imfs[n])
        plt.title(f'IMF {n + 1}')
        plt.legend()

    plt.tight_layout()
    plt.show()


    def plot_arima_predictions(Data, order, seasonal_order=None):
        model = SARIMAX(Data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        predictions = model_fit.forecast(steps=12)

        plt.figure(figsize=(10, 6))
        plt.plot(Data, label='Original')
        plt.plot(range(len(Data), len(Data) + 12), predictions, label='Predicted', linestyle='--')
        plt.title(f'ARIMA{order} {"x" + str(seasonal_order) if seasonal_order else ""}预测')
        plt.legend()
        plt.show()

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
    future_date = pd.date_range(start='2024-01', periods=36, freq='M')
    plt.figure(figsize=(12, 6))
    plt.plot(Data.index, Data['Price'], label='Historical Data')
    plt.plot(future_date, total_forecast, label='Forecast', linestyle='--')
    plt.title('长江成品油运价预测')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.savefig('./predict_fig/' + oil_type + '_' + start_port + '_' + end_port + '_' + load_level + '.png')
    plt.show()
