import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pylab import mpl
import os

import inspect

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

from invoke import task


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei']

# 数据
dates = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']
prices = [230, 230, 240, 250, 240, 215, 210, 205, 205, 205, 200, 200, 200, 230, 250, 220, 210, 180, 170, 185, 205, 200, 200, 200, 195, 200, 200, 220, 250, 260, 240, 235, 210, 200, 200, 200]

# 创建DataFrame
df = pd.DataFrame({'Date': dates, 'Price': prices})
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 绘制时序图
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Price'], marker='o')
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
    print("数据是稳定的")
else:
    print("数据是不稳定的")

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 准备数据
dates = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']
prices = [230, 230, 240, 250, 240, 215, 210, 205, 205, 205, 200, 200, 200, 230, 250, 220, 210, 180, 170, 185, 205, 200, 200, 200, 195, 200, 200, 220, 250, 260, 240, 235, 210, 200, 200, 200]

# 创建DataFrame
df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Price': prices})
train_data = torch.tensor(df['Price'][:24].values.astype(np.float32)).view(-1, 1)
test_data = torch.tensor(df['Price'][24:].values.astype(np.float32)).view(-1, 1)

# 构建神经网络模型
class BPNN(nn.Module):
    def __init__(self):
        super(BPNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.prelu(self.fc1(x))
        x = self.prelu(self.fc2(x))
        x = self.fc3(x)
        return x

model = BPNN()

# 训练模型
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
max_epochs = 50
for epoch in range(max_epochs):
    optimizer.zero_grad()
    outputs = model(train_data)
    loss = criterion(outputs, train_data)
    loss.backward()
    optimizer.step()

    if loss.item() < 0.001:
        break

# 预测
model.eval()
with torch.no_grad():
    predicted = model(test_data).data.numpy()

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], label='Original Data')
plt.plot(df['Date'][24:], predicted, label='Predicted Data', color='red')
plt.title('原始数据与BP神经网络预测数据对比')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.show()

# 预测整个数据集
model.eval()
with torch.no_grad():
    all_data = torch.tensor(df['Price'].values.astype(np.float32)).view(-1, 1)
    predicted_all = model(all_data).data.numpy()

print('打印:', type(predicted_all), predicted_all.flatten())

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], label='Original Data', marker='o')
plt.plot(df['Date'], predicted_all, label='Predicted Data', color='red', linestyle='dashed')
plt.title('原始数据与BP神经网络预测数据对比')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# 数据准备
dates = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']
prices = [230, 230, 240, 250, 240, 215, 210, 205, 205, 205, 200, 200, 200, 230, 250, 220, 210, 180, 170, 185, 205, 200, 200, 200, 195, 200, 200, 220, 250, 260, 240, 235, 210, 200, 200, 200]

# 初始化Kalman滤波器
kf = KalmanFilter(initial_state_mean=prices[0], n_dim_obs=1)

# 应用Kalman滤波器进行预测
measurements = np.array(prices)
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)

print('打印:', type(filtered_state_means.flatten()), filtered_state_means.flatten())

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(dates, prices, label='Original Data', marker='o')
plt.plot(dates, filtered_state_means.flatten(), label='Kalman Filter Prediction', color='red', linestyle='dashed')
plt.xticks(rotation=45)
plt.title('原始数据与Kalman滤波器预测数据对比')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.show()

import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pykalman import KalmanFilter
import os
from pylab import mpl

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['SimHei']
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 数据准备
dates = ['2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06', '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12']
prices = [230, 230, 240, 250, 240, 215, 210, 205, 205, 205, 200, 200, 200, 230, 250, 220, 210, 180, 170, 185, 205, 200, 200, 200, 195, 200, 200, 220, 250, 260, 240, 235, 210, 200, 200, 200]

df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Price': prices})
price_data = np.array(prices).reshape(-1, 1)

# 使用Kalman滤波器处理数据
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1, n_dim_state=1)
kf = kf.em(price_data, n_iter=5)
(filtered_state_means, _) = kf.filter(price_data)

# 构建BP神经网络
class BPNN(nn.Module):
    def __init__(self):
        super(BPNN, self).__init__()
        self.fc1 = nn.Linear(2, 10)  # 输入层接收原始价格和Kalman滤波器输出
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = BPNN()

# 训练数据准备
combined_data = np.hstack((price_data, filtered_state_means))  # 将原始数据和Kalman滤波器输出结合
combined_data_tensor = torch.tensor(combined_data, dtype=torch.float32)

# 训练模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(combined_data_tensor)
    loss = criterion(output, combined_data_tensor[:, 0:1])
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 预测
model.eval()
with torch.no_grad():
    predicted = model(combined_data_tensor).numpy()

print('打印:', type(predicted.flatten()), predicted.flatten())

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], prices, label='Original Data', marker='o')
plt.plot(df['Date'], predicted.flatten(), label='Kalman-BP Prediction', color='red', linestyle='dashed')
plt.xticks(rotation=45)
plt.title('原始数据与Kalman-BP组合模型预测数据对比')
plt.xlabel('日期')
plt.ylabel('价格')
plt.legend()
plt.show()

import numpy as np
import pandas as pd

# 假设这些是您的预测结果和原始数据
original_data = np.array(prices)
bp_predictions = predicted_all.flatten()  # BP预测结果
kalman_predictions = predicted_all.flatten()  # Kalman预测结果
kalman_bp_predictions = predicted.flatten()  # Kalman-BP预测结果

# 计算残差
bp_residuals = original_data - bp_predictions
kalman_residuals = original_data - kalman_predictions
kalman_bp_residuals = original_data - kalman_bp_predictions

# 计算MSE、MAE、MRE
def calculate_metrics(original, predicted):
    mse = np.mean((original - predicted) ** 2)
    mae = np.mean(np.abs(original - predicted))
    mre = np.mean(np.abs(original - predicted) / original)
    return mse, mae, mre

bp_mse, bp_mae, bp_mre = calculate_metrics(original_data, bp_predictions)
kalman_mse, kalman_mae, kalman_mre = calculate_metrics(original_data, kalman_predictions)
kalman_bp_mse, kalman_bp_mae, kalman_bp_mre = calculate_metrics(original_data, kalman_bp_predictions)

# 创建一个DataFrame来显示结果
results_df = pd.DataFrame({
    'Model': ['BP', 'Kalman', 'Kalman-BP'],
    'MSE': [bp_mse, kalman_mse, kalman_bp_mse],
    'MAE': [bp_mae, kalman_mae, kalman_bp_mae],
    'MRE': [bp_mre, kalman_mre, kalman_bp_mre]
})

print(results_df)


# 生成2024年的日期数据
future_dates = pd.date_range(start='2024-01-01', periods=12, freq='M')

# 由于我们没有2024年的真实输入数据，我们将使用2023年的数据作为基础来模拟2024年的输入
# 这里简化处理，实际应用中应根据具体情况生成或获取未来数据的预测输入
future_data = combined_data_tensor[-12:]  # 使用2023年的数据作为2024年数据的代理

# 使用训练好的模型进行预测
model.eval()
with torch.no_grad():
    future_predictions = model(future_data).numpy()

# 绘制历史数据和预测结果
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], prices, label='Historical Data', marker='o')
plt.plot(future_dates, future_predictions.flatten(), label='2024 Predictions', color='red', linestyle='--')
plt.title('Historical Data and 2024 Predictions with Kalman-BP Model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.show()
