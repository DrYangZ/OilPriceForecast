import math
import zipfile
from datetime import datetime

import openpyxl
import inspect
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pykalman import KalmanFilter
import os


if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif'] = ['SimHei']
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 数据准备


def main(out):
    dates = eval(out.split("*")[0])
    print('dates数据类型为：', type(dates))
    print(dates)
    prices = eval(out.split("*")[1])
    print('prices数据类型为：', type(prices))
    print(prices)
    date_time = out.split("*")[2]
    print('date_time数据类型为：', type(date_time))
    print(date_time)
    waterlevel1 = eval(out.split("*")[3])
    print('waterlevel1数据类型为：', type(waterlevel1))
    print(waterlevel1)
    waterlevel2 = eval(out.split("*")[4])
    prewaterlevel1 = eval(out.split("*")[5])
    print('prewaterlevel1数据类型为：', type(prewaterlevel1))
    print(prewaterlevel1)
    prewaterlevel2 = eval(out.split("*")[6])

    df = pd.DataFrame({'Date': pd.to_datetime(dates), 'Price': prices})
    price_data = np.array(prices).reshape(-1, 1)
    prewaterlevel1_data = np.array(prewaterlevel1).reshape(-1, 1)
    prewaterlevel2_data = np.array(prewaterlevel2).reshape(-1, 1)

    # 使用Kalman滤波器处理数据
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1, n_dim_state=1)
    kf = kf.em(price_data, n_iter=5)
    (filtered_state_means, _) = kf.filter(price_data)

    # class BPNN(nn.Module):
    #     def __init__(self):
    #         super(BPNN, self).__init__()
    #         self.fc1 = nn.Linear(1, 10)
    #         self.fc2 = nn.Linear(10, 10)
    #         self.fc3 = nn.Linear(10, 1)
    #         self.prelu = nn.PReLU()
    #
    #     def forward(self, x):
    #         x = self.prelu(self.fc1(x))
    #         x = self.prelu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x
    #
    # model = BPNN()
    # model.eval()
    # with torch.no_grad():
    #     all_data = torch.tensor(df['Price'].values.astype(np.float32)).view(-1, 1)
    #     predicted_all = model(all_data).data.numpy()


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
    # combined_data = np.hstack((price_data,prewaterlevel1_data,prewaterlevel2_data, filtered_state_means))  # 将原始数据和Kalman滤波器输出结合
    combined_data = np.hstack((price_data, filtered_state_means))  # 将原始数据和Kalman滤波器输出结合
    combined_data_tensor = torch.tensor(combined_data, dtype=torch.float32)

    # 训练模型
    learn_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    criterion = nn.MSELoss()

    for epoch in range(200):
        optimizer.zero_grad()
        output = model(combined_data_tensor)
        loss = criterion(output, combined_data_tensor[:, 0:1])
        loss.backward()
        optimizer.step()
        optimizer.lr = (loss.item()/10) * learn_rate
        if 1000 < loss.item() < 5000:
            optimizer.lr = (loss.item()/10) * learn_rate * 0.8
        elif 150 < loss.item() < 1000:
            optimizer.lr = (loss.item()/10) * learn_rate * 0.7
        elif loss.item() < 150:
            optimizer.lr = loss.item() * learn_rate * 0.1
        # elif loss.item() < 15:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.8
        if epoch % 5 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # 预测
    model.eval()
    with torch.no_grad():
        predicted = model(combined_data_tensor).numpy()

    #预测值
    out_pre = list(predicted.flatten())
    # print(out_pre)

    # print('打印:', type(predicted.flatten()), predicted.flatten())

    # 绘制图表
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], prices, label='Original Data', marker='o')
    plt.plot(df['Date'], predicted.flatten(), label='Kalman-BP Prediction', color='red', linestyle='dashed')
    plt.xticks(rotation=45)
    plt.title('原始数据与Kalman-BP组合模型预测数据对比')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    # plt.show()
    plt.savefig("output"+ "\\" + "out_pre.png")



    # 假设这些是您的预测结果和原始数据
    original_data = np.array(prices)
    bp_predictions = predicted.flatten()  # BP预测结果
    kalman_predictions = predicted.flatten()  # Kalman预测结果
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
        'Model': ['BP', 'Kalman', 'Combined'],
        'MSE': [round(bp_mse,1), round(kalman_mse,1), round(kalman_bp_mse,1)],
        'MAE': [round(bp_mae,1), round(kalman_mae,1), round(kalman_bp_mae,1)],
        'MRE': [round(bp_mre,1), round(kalman_mre,1), round(kalman_bp_mre,1)]
    })

    print(results_df)

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    for index, value in enumerate(out_pre, start=1):
        sheet.cell(row=index, column=1).value = value
    workbook.save("output\\out.xlsx")

    writer = "output\\out_parameter.xlsx"
    results_df.to_excel(writer, index=True, sheet_name='Sheet2')

    with open("out.txt", "w+") as f:
        for i, pri in enumerate(out_pre):
            f.write(str(pri))
            f.write("   ")
            if ((i + 1) % 3) == 0:
                f.write("\r\n")

        f.write('\r\n')
        content2 = str(results_df)
        f.write(content2)





    # 生成2024年的日期数据
    future_dates = pd.date_range(start='2024-01-01', periods=12, freq='ME')

    # 由于我们没有2024年的真实输入数据，我们将使用2023年的数据作为基础来模拟2024年的输入
    # 这里简化处理，实际应用中应根据具体情况生成或获取未来数据的预测输入
    future_data = combined_data_tensor[-12:]  # 使用2023年的数据作为2024年数据的代理

    # 使用训练好的模型进行预测
    model.eval()
    with torch.no_grad():
        future_predictions = model(future_data).numpy()



    price_predictout = [i*((math.atan(waterlevel2*0.3 + waterlevel1*0.7)/1.57)*0.4 + 0.8) for i in future_predictions.flatten()]



    #生成未来的时间列表
    start_date = datetime.strptime(date_time, '%Y-%m')
    date_list = []
    for i in range(12):
        new_date = start_date.replace(year=start_date.year + (start_date.month + i -1) // 12,
                                      month=(start_date.month + i - 1) % 12 + 1)
        date_list.append(new_date.strftime('%Y-%m'))





    # 绘制历史数据和预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(dates, prices, label='实际值')
    plt.plot(date_list[0], price_predictout[0],marker='*', label='组合模型Kalman-BP预测')

    # plt.plot(date, prices, label='Historical Data', marker='o')
    # plt.plot(date_list, future_predictions.flatten(), label='2024 Predictions', color='red', linestyle='--')
    plt.title('历史数据与预测数据')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    # plt.show()
    plt.savefig("output" + "\\" + "nextyear_pre.png")


    #保存预测的值
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

