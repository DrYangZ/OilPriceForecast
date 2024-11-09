# -*- coding: utf-8 -*-
import os.path
import sys
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pykalman import KalmanFilter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 设置标准输出的编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# 读取数据集
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name)
    df['时间'] = pd.to_datetime(df['时间'], format='%Y-%m-%d')
    df['月份'] = df['时间'].dt.strftime('%m').astype(int)
    df.set_index('时间', inplace=True)

    features = df[
        ['月份',
         '成品油运价指数',
         '汽油价格（元/吨）',
         '柴油价格（元/吨）月底价格',
         '水位（m）',
         '水上运输业从业人员数',
         '管道运输业从业人员数',
         'GDP指数',
         'CPI',
         '国家财政收入（居民消费价格指数）',
         '能源消费总量',
         '石油能源消费总量(万吨)',
         '内河港口石油天然气进出港吞吐量',
         '成品油消费量',
         '运力保有量',
         '管道运输规模']]
    target = df[['价格']]
    return features, target, df

# 应用卡尔曼滤波器对特征数据进行平滑处理
def apply_kalman_filter(features):
    kf = KalmanFilter(initial_state_mean=np.mean(features, axis=0), n_dim_obs=features.shape[1])
    filtered_state_means, _ = kf.filter(features)
    return filtered_state_means

# 构建BP神经网络
def build_bp_model(input_shape):
    model = Sequential()
    model.add(Dense(256, input_dim=input_shape, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# 进行归一化和反归一化
def scale_data(features, target):
    scaler_features = MinMaxScaler()
    scaled_features = scaler_features.fit_transform(features)

    scaler_target = MinMaxScaler(feature_range=(0.2, 0.8))
    scaled_target = scaler_target.fit_transform(target)

    return scaled_features, scaled_target, scaler_features, scaler_target

# 预测未来价格的函数
def predict_future(bp_model, last_known_input, future_steps, scaler_features, scaler_target):
    future_predictions = []

    current_input = last_known_input  # 初始化为最后已知的输入特征

    for _ in range(future_steps):
        # 预测下一步
        next_pred_scaled = bp_model.predict(current_input.reshape(1, -1))  # 使用已训练好的模型预测
        next_pred = scaler_target.inverse_transform(next_pred_scaled)  # 反归一化预测的价格

        future_predictions.append(next_pred[0, 0])  # 存储预测结果

        # 更新输入，模拟未来的特征
        current_input = np.roll(current_input, -1)  # 滚动特征，将新的预测价格作为下一个输入
        current_input[-1] = next_pred_scaled[0][0]  # 用预测的值作为下一个输入的最后一个特征值

    return np.array(future_predictions)

# 主函数流程
def main(file_path='./Datasets.xlsx',sheet_name='南京到重庆-贸易商', forecast_steps=6):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 读取数据集
    features, target, df = load_data(file_path, sheet_name)
    # 2. 对特征值应用卡尔曼滤波器进行平滑
    kalman_features = apply_kalman_filter(features.values)
    # 3. 对特征值和目标值进行归一化
    scaled_features, scaled_target, scaler_features, scaler_target = scale_data(kalman_features, target.values)
    # 4. 构建BP神经网络
    bp_model = build_bp_model(scaled_features.shape[1])
    # 5. 增加EarlyStopping，避免过拟合
    early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
    if not os.path.exists('./bp.weights.h5'):
        # 6. 训练模型
        bp_model.fit(scaled_features, scaled_target, epochs=500, batch_size=16, verbose=1, callbacks=[early_stopping])
        bp_model.save_weights('./weights/bp.weights.h5')
        print("模型权重已保存到 './weights/bp.weights.h5'")
    else:
        # 加载模型权重
        bp_model.load_weights('./weights/bp.weights.h5')
        print("模型权重已成功加载")

    # 7. 使用模型进行预测
    predictions = bp_model.predict(scaled_features)
    # 8. 反归一化预测结果
    predictions_rescaled = scaler_target.inverse_transform(predictions)
    target_rescaled = scaler_target.inverse_transform(scaled_target)
    # 9. 评估模型性能
    mse = mean_squared_error(target_rescaled, predictions_rescaled)
    r2 = r2_score(target_rescaled, predictions_rescaled)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # # 10. 可视化训练结果
    # plt.plot(target_rescaled, label='真实价格')
    # plt.plot(predictions_rescaled, label='预测价格', color='red')
    # plt.legend()
    # plt.show()

    # 11. 预测未来的价格
    target_price = []
    last_known_input = scaled_features[-1, :]  # 获取最后一个已知的特征作为初始输入
    future_pred = predict_future(bp_model, last_known_input, forecast_steps, scaler_features, scaler_target)
    target_price.append(df['价格'].iloc[-1])
    target_price.extend(future_pred)
    print(target_price)
    print(f'未来 {forecast_steps} 个月的预测价格为: {future_pred}')

    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')  # 使用ggplot风格

    # 绘制原始价格序列
    plt.plot(range(len(df)), df['价格'], label='原始价格', lw=2, color='#1F4E79')
    # 绘制预测序列
    plt.plot(range(len(df) - 1, len(df) + forecast_steps), target_price,
             label='预测价格', color='#e74c3c', linestyle='--', lw=2)
    # 设置标题和标签
    plt.title('Kalman-BP预测结果', fontsize=16, fontweight='bold')
    plt.xlabel('月份', fontsize=14)
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

    plt.savefig('./fig/Kalman-BP.png')
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
    save_path = './predict_data/Kalman-BP.csv'
    display_data.to_csv(save_path, index=False)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1:]
        print(f'EMD-ARIMA.py script has received arguments values from sys.argv{arg}')
        main(arg[0], arg[1], int(arg[2]))
    # main()
