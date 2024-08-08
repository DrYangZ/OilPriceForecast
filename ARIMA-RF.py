import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from pylab import mpl
from statsmodels.graphics.gofplots import qqplot

plt.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei']

# 数据
dates = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']
prices = [230, 230, 240, 250, 240, 215, 210, 205, 205, 205, 200, 200, 200, 230, 250, 220, 210, 180, 170, 185, 205, 200, 200, 200, 195, 200, 200, 220, 250, 260, 240, 235, 210, 200, 200, 200]

# 将数据转换成pandas DataFrame
data = pd.DataFrame({'Date': dates, 'Price': prices})
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 绘制时序图
plt.figure(figsize=(12, 6))
plt.plot(data['Price'], marker='o')
plt.title('长江成品油运价时序图')
plt.xlabel('日期')
plt.ylabel('价格')
plt.grid(True)
plt.show()

# 进行ADF检验
result = adfuller(data['Price'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# 判断数据的稳定性
if result[0] < result[4]["5%"]:
    print("数据是平稳的")
else:
    print("数据是非平稳的")

# 对Price进行一阶差分
data['Price_Diff'] = data['Price'].diff()

# 去除NaN值
data_diff = data.dropna()

# 绘制一阶差分后的时序图
plt.figure(figsize=(12, 6))
plt.plot(data_diff['Price_Diff'], marker='o')
plt.title('长江成品油运价一阶差分时序图')
plt.xlabel('日期')
plt.ylabel('价格差分')
plt.grid(True)
plt.show()

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

# 绘制ACF图
plt.figure(figsize=(12, 6))
plot_acf(data_diff['Price_Diff'], lags=20)
plt.title('一阶差分序列的自相关系数 (ACF)')
plt.show()

# 绘制PACF图
plt.figure(figsize=(12, 6))
plot_pacf(data_diff['Price_Diff'], lags=16)
plt.title('一阶差分序列的偏自相关系数 (PACF)')
plt.show()

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

# 绘制预测数据的时序图
plt.figure(figsize=(12, 6))
plt.plot(data['Price'], label='Actual')
plt.plot(data['Forecast'], label='Forecast')
plt.title('实际数据与ARIMA(2,1,2)预测数据的时序图')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.show()

# 绘制残差序列的时序图
plt.figure(figsize=(12, 6))
plt.plot(data['Residual'], label='Residual')
plt.title('残差序列的时序图')
plt.xlabel('日期')
plt.ylabel('残差')
plt.legend()
plt.grid(True)
plt.show()

# 绘制残差序列的ACF图
plt.figure(figsize=(12, 6))
plot_acf(data['Residual'].dropna(), lags=20)
plt.title('残差序列的自相关系数 (ACF)')
plt.show()

# 绘制残差序列的PACF图
plt.figure(figsize=(12, 6))
plot_pacf(data['Residual'].dropna(), lags=17)
plt.title('残差序列的偏自相关系数 (PACF)')
plt.show()

# 绘制残差序列的QQ图
plt.figure(figsize=(8, 8))
qqplot(data['Residual'].dropna(), line='s')
plt.title('残差序列的QQ图')
plt.show()

# 打印实际数据、预测数据、对应误差的表格
print(data[['Price', 'Forecast', 'Residual']])

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


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

# 绘制时序图
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], y, label='Actual')
plt.plot(data['Date'], y_pred, label='RF_Predicted')
plt.title('实际与预测价格的时序图')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.xticks(rotation=90)
plt.show()

print(type(y_pred))

combined_pred = []
# 计算组合预测结果
for i, j in zip(y_pred_arima_list, y_pred_rf_list):
    combined_data = 0.88 * i + 0.12 * j
    combined_pred.append(combined_data)
date_list = data['Date'].tolist()

# 绘制时序图
plt.figure(figsize=(12, 6))
plt.plot(date_list, y, label='Actual')
plt.plot(date_list, combined_pred, label='Combined Predicted')
plt.title('实际数据与组合预测结果的时序图')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.show()

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
plt.show()

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
    'Model': ['ARIMA', 'RF', 'ARIMA-RF Combined'],
    'MSE': [mse_arima, mse_rf, mse_combined],
    'RMSE': [rmse_arima, rmse_rf, rmse_combined],
    'MAPE': [mape_arima, mape_rf, mape_combined]
})

# 打印表格
print(metrics_df)


import pandas as pd
from pandas.tseries.offsets import MonthEnd

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

# 绘制未来一年的预测数据
plt.figure(figsize=(12, 6))
plt.plot(future_dates_df['Date'], future_pred_df['Combined_Pred'], label='Combined Predicted')
plt.title('后一年的价格预测')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.grid(True)
plt.xticks(rotation=90)
plt.show()
