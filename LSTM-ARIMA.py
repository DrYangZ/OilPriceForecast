import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pylab import mpl
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei']

# 数据
dates = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08',
         '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04',
         '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
         '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08',
         '2023-09', '2023-10', '2023-11', '2023-12']
prices = [230, 230, 240, 250, 240, 215, 210, 205, 205, 205, 200, 200, 200, 230, 250, 220, 210,
          180, 170, 185, 205, 200, 200, 200, 195, 200, 200, 220, 250, 260, 240, 235, 210, 200, 200, 200]

# 创建DataFrame
df = pd.DataFrame({'Date': dates, 'Price': prices})
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 绘制时序图
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Price'])
plt.title('长江成品油运价时序图')
plt.xlabel('日期')
plt.ylabel('价格')
plt.grid(True)
plt.show()

# ADF检验
result = adfuller(df['Price'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

# 判断稳定性
if result[0] < result[4]["5%"]:
    print("数据是稳定的。")
else:
    print("数据不是稳定的。")

# 时间序列分解
decomposition = seasonal_decompose(df['Price'], model='additive', period=12)

# 获取趋势（线性成分）
trend = decomposition.trend

# 绘制分解结果
plt.figure(figsize=(14, 8))

plt.subplot(411)
plt.plot(df['Price'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 移除NaN值
trend.dropna(inplace=True)

# 二阶差分
trend_diff = trend.diff().diff().dropna()

# 绘制ACF和PACF图
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(trend_diff, ax=plt.gca(), lags=11)
plt.subplot(212)
plot_pacf(trend_diff, ax=plt.gca(), lags=10)
plt.tight_layout()
plt.show()

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# 使用ARIMA(1,2,2)模型
model = ARIMA(df['Price'], order=(1, 2, 2))
model_fit = model.fit()

# 进行预测
df['forecast'] = model_fit.predict(start=0, end=len(df) - 1, typ='levels')
df['forecast'][0] = df['Price'][0]

data_pred_arima_list = [i for i in df['forecast'].values]
print('打印:', type(data_pred_arima_list), data_pred_arima_list)

# 计算残差
df['residual'] = df['Price'] - df['forecast']

# Ljung-Box检验
lb_test = sm.stats.acorr_ljungbox(df['residual'].dropna(), lags=[10], return_df=True)
print(lb_test)

# 绘制ACF和PACF图
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(df['residual'].dropna(), ax=plt.gca())
plt.subplot(212)
plot_pacf(df['residual'].dropna(), ax=plt.gca())
plt.tight_layout()
plt.show()

# 判断残差是否符合白噪声
if lb_test['lb_pvalue'][10] > 0.05:
    print("残差序列是白噪声，模型拟合良好。")
else:
    print("残差序列不是白噪声，模型可能需要进一步优化。")


import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 数据预处理
data = pd.DataFrame(prices, columns=['Price'])
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 创建数据集
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
X, Y = create_dataset(data_scaled, look_back)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 调整学习率
from keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    if epoch % 400 == 0 and epoch != 0:
        return lr * 0.2
    return lr

callback = LearningRateScheduler(scheduler)

# 模型训练
model.fit(X, Y, epochs=1000, batch_size=1, verbose=1, callbacks=[callback])

# 进行预测
predictions = model.predict(X)

# 反转缩放
predictions = scaler.inverse_transform(predictions)
Y = scaler.inverse_transform([Y])
data_pred_lstm_list = predictions[:,0].tolist()
data_pred_lstm_list.append(prices[-1])
print('打印:', type(predictions[:,0]), predictions[:,0])

# 绘制时序图
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Price'], label='Original')
plt.plot(data.index[look_back:], predictions[:,0], label='Predicted')
plt.title('原始数据与LSTM预测')
plt.xlabel('时间')
plt.ylabel('价格')
plt.legend()
plt.show()


# 假设arima_predictions和lstm_predictions是我们之前获得的预测结果
# actual_values代表实际的时间序列数据
# 这些数据应该是等长的

arima_predictions = data_pred_arima_list  # ARIMA模型的预测值
lstm_predictions = data_pred_lstm_list   # LSTM模型的预测值
actual_values = data['Price'].values.tolist()      # 实际值

arima_errors = []
lstm_errors = []
# 计算预测误差
for i, j, z in zip(arima_predictions, lstm_predictions, actual_values):
    arima_error = np.abs(i - z)
    lstm_error = np.abs(j - z)
    arima_errors.append(arima_error)
    lstm_errors.append(lstm_error)

# 计算每个模型的平均误差
arima_mean_error = np.mean(arima_errors)
lstm_mean_error = np.mean(lstm_errors)

# 计算权重（预测误差越小，权重越大）
total_error = arima_mean_error + lstm_mean_error
arima_weight = (1 - arima_mean_error / total_error)
lstm_weight = (1 - lstm_mean_error / total_error)

# 计算组合模型的预测值
combined_predictions = []
for i, j in zip(arima_predictions, lstm_predictions):
    combined_prediction = arima_weight * i + lstm_weight * j
    combined_predictions.append(combined_prediction)

# 绘制预测结果与实际数据
plt.figure(figsize=(12, 6))
plt.plot(actual_values, label='实际值')
plt.plot(arima_predictions, label='ARIMA预测')
plt.plot(lstm_predictions, label='LSTM预测')
plt.plot(combined_predictions, label='组合模型预测')
plt.title('预测结果与实际数据比较')
plt.xlabel('时间')
plt.ylabel('数值')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 定义MAPE计算函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 计算R2, RMSE, MAPE, MAE
metrics = pd.DataFrame(index=['R2', 'RMSE', 'MAPE', 'MAE'],
                       columns=['ARIMA', 'LSTM', 'Combined'])

for name, predictions in zip(['ARIMA', 'LSTM', 'Combined'],
                             [arima_predictions, lstm_predictions, combined_predictions]):
    metrics.loc['R2', name] = r2_score(actual_values, predictions)
    metrics.loc['RMSE', name] = np.sqrt(mean_squared_error(actual_values, predictions))
    metrics.loc['MAPE', name] = mean_absolute_percentage_error(actual_values, predictions)
    metrics.loc['MAE', name] = mean_absolute_error(actual_values, predictions)

# 打印表格
print(metrics)




# 使用ARIMA模型进行预测
# 假设周期性为12个月，模型参数为(1,2,2)，基于给定数据
model = ARIMA(df['Price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# 预测未来12个月
arima_pred_list_2024 = model_fit.forecast(steps=12)

# 打印预测结果
print(arima_pred_list_2024)
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Price'], label='Original Data')
plt.plot(pd.date_range(df.index[-1] + pd.Timedelta(days=31), periods=12, freq='M'), arima_pred_list_2024, label='Forecast', linestyle='--')
plt.title('Price Forecast for 2024')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# 数据准备
dates = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08',
         '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04',
         '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
         '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08',
         '2023-09', '2023-10', '2023-11', '2023-12']
prices = [230, 230, 240, 250, 240, 215, 210, 205, 205, 205, 200, 200, 200, 230, 250, 220, 210,
          180, 170, 185, 205, 200, 200, 200, 195, 200, 200, 220, 250, 260, 240, 235, 210, 200, 200, 200]

# 数据预处理
prices = np.array(prices).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# 生成时间序列数据
n_input = 12
n_features = 1
generator = TimeseriesGenerator(prices_scaled, prices_scaled, length=n_input, batch_size=6)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(generator, epochs=200, verbose=1)

# 预测未来一年的数据
lstm_pred_list_2024 = []
batch = prices_scaled[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):
    lstm_pred_list_2024.append(model.predict(batch)[0])
    batch = np.append(batch[:,1:,:],[[lstm_pred_list_2024[i]]],axis=1)

# 反标准化预测结果
lstm_pred_list_2024 = scaler.inverse_transform(lstm_pred_list_2024)

# 绘制结果
predicted_dates = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06', '2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12']
plt.figure(figsize=(10,6))
plt.plot(dates, prices, label='Actual Prices')
plt.plot(predicted_dates, lstm_pred_list_2024, label='Predicted Prices')
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Prediction using LSTM')
plt.legend()
plt.show()


# 计算组合模型的预测值
arima_weight = 0.5
lstm_weight = 0.6
combined_predictions_2024 = []
for i, j in zip(arima_pred_list_2024, lstm_pred_list_2024):
    combined_prediction = arima_weight * i + lstm_weight * j
    combined_predictions_2024.append(combined_prediction)

# 绘制预测结果与实际数据
plt.figure(figsize=(12, 6))
plt.plot(dates, actual_values, label='实际值')
plt.plot(predicted_dates, combined_predictions_2024, label='组合模型预测2024')
plt.title('预测结果与实际数据比较')
plt.xlabel('时间')
plt.ylabel('数值')
plt.legend()
plt.show()
