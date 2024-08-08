import math
import os
import zipfile
from datetime import datetime

import matplotlib.pyplot as plt
import openpyxl
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pylab import mpl
from statsmodels.graphics.gofplots import qqplot
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif'] = ['SimHei']




def main(Date, Price, date_time, waterlevel1, waterlevel2):
    # 数据
    dates = Date
    prices = Price


    # 将数据转换成pandas DataFrame
    data = pd.DataFrame({'Date': dates, 'Price': prices})
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # 对Price进行一阶差分
    data['Price_Diff'] = data['Price'].diff()

    # 去除NaN值
    data_diff = data.dropna()

    # 对一阶差分序列进行ADF检验
    result_diff = adfuller(data_diff['Price_Diff'])
    print('ADF Statistic for Differenced Series: %f' % result_diff[0])
    print('p-value: %f' % result_diff[1])
    print('Critical Values:')
    for key, value in result_diff[4].items():
        print('\t%s: %.3f' % (key, value))

    # 判断差分序列的稳定性
    if result_diff[0] < result_diff[4]["5%"]:
        print("差分序列是平稳的")
    else:
        print("差分序列是非平稳的")


    # 使用ARIMA(2,1,2)模型进行拟合
    model = ARIMA(data['Price'], order=(2, 1, 2))
    model_fit = model.fit()

    # 进行预测
    data['Forecast'] = model_fit.predict(start=0, end=len(data) - 1, typ='levels')
    data['Forecast'][0] = data['Price'][0]

    print('打印：', data['Forecast'].values)
    y_pred_arima_list = data['Forecast'].tolist()
    y_pred_arima_list = [float(i) for i in y_pred_arima_list]

    print(type(data['Forecast']))

    # 计算残差
    data['Residual'] = data['Price'] - data['Forecast']

    # 打印实际数据、预测数据、对应误差的表格
    print(data[['Price', 'Forecast', 'Residual']])




    # 生成模拟影响因子数据
    np.random.seed(0)
    factor1 = np.random.normal(0, 1, 36)
    factor2 = np.random.normal(0, 1, 36)

    # 创建DataFrame
    data = pd.DataFrame({'Date': dates, 'Price': prices, 'Factor1': factor1, 'Factor2': factor2})

    # 分割数据集
    X = data[['Factor1', 'Factor2']]
    y = data['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # 构建随机森林模型
    rf = RandomForestRegressor(random_state=0)

    # 设置网格搜索的参数范围
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20, 30],
    }

    # 进行网格搜索
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 使用最佳参数的模型进行预测
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X)

    print('打印:', y_pred)
    y_pred_rf_list = y_pred.tolist()
    y_pred_rf_list = [float(i) for i in y_pred_rf_list]

    # 计算MSE, RMSE, MAPE
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y, y_pred)

    # 打印评价指标
    print(f'MSE: {mse}, RMSE: {rmse}, MAPE: {mape}')



    print(type(y_pred))

    combined_pred = []
    # 计算组合预测结果
    for i, j in zip(y_pred_arima_list, y_pred_rf_list):
        combined_data = 0.88 * i + 0.12 * j
        combined_pred.append(combined_data)
    date_list = data['Date'].tolist()



    # 计算评价指标
    mse_combined = mean_squared_error(y, combined_pred)
    rmse_combined = np.sqrt(mse_combined)
    mape_combined = mean_absolute_percentage_error(y, combined_pred)

    # 打印评价指标
    print(f'Combined Model MSE: {mse_combined}, RMSE: {rmse_combined}, MAPE: {mape_combined}')

    # y: 实际数据
    # y_pred_arima: ARIMA模型的预测结果
    # y_pred_rf: RF模型的预测结果
    # combined_pred: ARIMA-RF组合预测结果

    # 绘制时序图
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], y, label='Actual', color='blue')
    plt.plot(data['Date'], y_pred_arima_list, label='ARIMA Predicted', color='red')
    plt.plot(data['Date'], y_pred_rf_list, label='RF Predicted', color='green')
    plt.plot(data['Date'], combined_pred, label='Combined Predicted', color='purple')
    plt.title('实际数据与各模型预测结果的时序图')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.savefig("output" + "\\" + "out_pre.png")

    # 计算评价指标
    mse_arima = mean_squared_error(y, y_pred_arima_list)
    rmse_arima = np.sqrt(mse_arima)
    mape_arima = mean_absolute_percentage_error(y, y_pred_arima_list)

    mse_rf = mean_squared_error(y, y_pred_rf_list)
    rmse_rf = np.sqrt(mse_rf)
    mape_rf = mean_absolute_percentage_error(y, y_pred_rf_list)

    mse_combined = mean_squared_error(y, combined_pred)
    rmse_combined = np.sqrt(mse_combined)
    mape_combined = mean_absolute_percentage_error(y, combined_pred)

    # 创建评估指标表格
    metrics_df = pd.DataFrame({
        'Model': ['ARIMA', 'RF', 'Combined'],
        'MSE': [round(mse_arima,1), round(mse_rf,1), round(mse_combined,1)],
        'RMSE': [round(rmse_arima,1), round(rmse_rf,1), round(rmse_combined,1)],
        'MAPE': [round(mape_arima,1), round(mape_rf,1), round(mape_combined,1)]
    })

    # 打印表格
    print(metrics_df)



    workbook = openpyxl.Workbook()
    sheet = workbook.active
    for index, value in enumerate(y_pred_rf_list, start=1):
        sheet.cell(row=index, column=1).value = value
    workbook.save("output\\out.xlsx")



    #将参数写入表格中
    writer = "output\\out_parameter.xlsx"
    metrics_df.to_excel(writer, index=True, sheet_name='Sheet2')

    with open("out.txt", "w+") as f:
        for i, pri in enumerate(y_pred_rf_list):
            f.write(str(pri))
            f.write("   ")
            if ((i + 1) % 3) == 0:
                f.write("\r\n")

        f.write('\r\n')
        content2 = str(metrics_df)
        f.write(content2)



    # 假设已有的最后一个日期是2023-12
    last_date = pd.to_datetime('2023-12')

    # 生成后一年的日期列表
    future_dates = [last_date + MonthEnd(n) for n in range(1, 13)]
    future_dates_df = pd.DataFrame({'Date': future_dates})

    # 预测后一年的数据
    future_arima_pred = model_fit.forecast(steps=12)  # ARIMA模型预测后一年
    future_rf_pred = np.array([np.mean(y_pred_rf_list[-12:]) for _ in range(12)])  # 使用RF模型最后12个月的平均值作为简单预测

    # 组合模型的预测
    future_combined_pred = 0.88 * future_arima_pred + 0.12 * future_rf_pred

    # 创建预测数据的DataFrame
    future_pred_df = pd.DataFrame({
        'Date': future_dates,
        'ARIMA_Pred': future_arima_pred,
        'RF_Pred': future_rf_pred,
        'Combined_Pred': future_combined_pred
    })

    start_date = datetime.strptime(date_time, '%Y-%m')
    date_list = []
    for i in range(12):
        new_date = start_date.replace(year=start_date.year + (start_date.month + i - 1) // 12,
                                      month=(start_date.month + i - 1) % 12 + 1)
        date_list.append(new_date.strftime('%Y-%m'))
    # print(date_list)
    # 绘制未来一年的预测数据


    price_predictout = [i*((math.atan(waterlevel2*0.3 + waterlevel1*0.7)/1.57)*0.4 + 0.8) for i in list(future_pred_df['Combined_Pred'])]




    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, label='实际值')
    plt.plot(date_list[0], price_predictout[0], marker='*',label='Combined Predicted')
    plt.xticks(rotation=45)
    plt.xticks(fontsize=10)
    plt.title('后一年的价格预测')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.savefig("output" + "\\" + "nextyear_pre.png")




    # 保存预测的值
    data_series = pd.Series([price_predictout[0]], index=[date_list[0]], name='Data')
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
