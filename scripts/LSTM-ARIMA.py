# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import os
import sys
import io
import warnings
from collections import deque

# 设置标准输出的编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
def create_sequences(features, target, time_steps):
    X, y = [], []

    for i in range(len(features) - time_steps):
        X.append(features[i:i + time_steps])
        y.append(target[i + time_steps])

    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(30, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(30)))
    model.add(Dropout(0.2))
    model.add(Dense(15))
    model.add(Dense(1))  # 输出层
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def predict_future(model, data, time_steps, future_steps):
    predictions = []
    current_input = data[-time_steps:]
    current_input = current_input.reshape(1, time_steps, data.shape[1])

    for _ in range(future_steps):
        next_pred = model.predict(current_input)
        predictions.append(next_pred[0, 0])

        new_input = np.copy(current_input)
        new_input = np.roll(new_input, -1, axis=1)
        new_input[0, -1, -1] = next_pred[0, 0]  # 提取标量值并更新目标值

        current_input = new_input

    return np.array(predictions)


def main(file_path='./Datasets.xlsx', sheet_name='南京到重庆-贸易商', forecast_steps=6):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore', category=UserWarning)

    df = pd.read_excel(file_path, sheet_name)
    df['时间'] = pd.to_datetime(df['时间'], format='%Y-%m-%d')
    df['月份'] = df['时间'].dt.strftime('%m').astype(int)
    df.set_index(df['时间'], inplace=True)
    predict_img_path = './fig/LSTM-ARIMA.png'

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
         '管道运输规模']
    ]
    target = df[['价格']]

    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(features)

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target)

    time_steps = 3
    X, y = create_sequences(scaled_features, scaled_target, time_steps)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)

    weights_path = './lstm.weights.h5'
    if not os.path.exists(weights_path):
        model.fit(X_train, y_train, epochs=100, batch_size=16)
        # 保存模型权重
        model.save_weights('./weights/lstm.weights.h5')
        print("模型权重已保存到 'lstm_model_weights.h5'")
    else:
        # 加载模型权重
        model.load_weights('./weights/lstm.weights.h5')
        print("模型权重已成功加载")

    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler_target.inverse_transform(y_pred)
    y_test_rescaled = scaler_target.inverse_transform(y_test)

    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    # plt.plot(y_test_rescaled, label='真实价格')
    # plt.plot(y_pred_rescaled, label='预测价格', color='red')
    # plt.legend()
    # plt.show()

    target_price = deque()
    future_pred_scaled = predict_future(model, scaled_features, time_steps, forecast_steps)
    future_pred_rescaled = scaler_target.inverse_transform(future_pred_scaled.reshape(-1, 1))
    predict_price = future_pred_rescaled.flatten()
    print(f'未来 {forecast_steps} 个月的预测价格为: {future_pred_rescaled.flatten()}')
    target_price.append(df['价格'].iloc[-1])
    target_price.extend(predict_price.tolist())
    print(target_price)
    print(predict_price.shape)
    print(type(predict_price))

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
        plt.scatter(len(df) + forecast_steps, target_price, color='red', marker='o', s=100,
                    label=f'价格的最终预测点')
    # 设置标题和标签
    plt.title('LSTM-ARIMA模型预测结果', fontsize=16, fontweight='bold')
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

    plt.savefig(predict_img_path)
    # plt.show()
    plt.close()
    result = list(target_price)[1:]
    predict_date = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_steps,
                                 freq='ME').strftime('%Y-%m').values
    display_data = pd.DataFrame({
        '时间': predict_date,
        '价格': result
    })
    print(display_data)
    save_path = './predict_data/LSTM-ARIMA.csv'
    display_data.to_csv(save_path, index=False)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1:]
        print(f'EMD-ARIMA.py script has received arguments values from sys.argv{arg}')
        main(arg[0], arg[1], int(arg[2]))
    # main()