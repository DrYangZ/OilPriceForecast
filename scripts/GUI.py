# -*- coding: utf-8 -*-
import re
import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLineEdit, QWidget, QMessageBox,
    QComboBox, QCheckBox, QLabel, QHBoxLayout, QVBoxLayout, QScrollArea, QDialog,
    QSplitter, QFrame, QTextEdit, QTableWidget, QTableWidgetItem, QFileDialog
)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QProcess
import sqlite3

# 设置 QT_PLUGIN_PATH，以确保找到所有 Qt 插件
os.environ["QT_PLUGIN_PATH"] = os.path.join(os.getcwd(), r".\python310\Lib\site-packages\PyQt5\Qt5\plugins")
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 设置窗口图标
        self.setWindowIcon(QIcon(r'./fig/logo_HD.jpg'))
        # 默认图片路径
        self.default_image_path = r"./fig/default.jpg"
        self.model_name = ['EMD-ARIMA', 'RF-ARIMA', 'LSTM-ARIMA', 'Kalman-BP']
        self.img_path = {
            r'EMD-ARIMA': './fig/EMD-ARIMA.png',
            r'RF-ARIMA': './fig/RF-ARIMA.png',
            r'LSTM-ARIMA': './fig/LSTM-ARIMA.png',
            r'Kalman-BP': './fig/Kalman-BP.png',
        }
        self.scripts = {
            r'EMD-ARIMA': './EMD-ARIMA.py',
            r'RF-ARIMA': './RF-ARIMA.py',
            r'LSTM-ARIMA': './LSTM-ARIMA.py',
            r'Kalman-BP': './Kalman-BP.py',
        }
        self.predict_data = {
            r'EMD-ARIMA': './predict_data/EMD-ARIMA.csv',
            r'RF-ARIMA': './predict_data/RF-ARIMA.csv',
            r'LSTM-ARIMA': './predict_data/LSTM-ARIMA.csv',
            r'Kalman-BP': './predict_data/Kalman-BP.csv',
        }
        # 获取指定的宽度和高度
        self.img_width = 1000
        self.img_height = 700
        self.process = None
        self.file_path = None
        self.file_path = None
        self.initUI()

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('长江成品油运价预测')
        self.setGeometry(0, 0, 1920, 1080)
        # # 设置窗口的背景颜色为纯白色
        self.setStyleSheet("background-color: white;")

        # 创建一个水平布局
        splitter = QSplitter()
        splitter.setStyleSheet("""
            background-color: white;
        """)

        # 创建三个垂直布局，并添加到分割器中
        left_layout = QVBoxLayout()
        center_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # 创建 QFrame 并设置样式表
        left_frame = QFrame()
        left_frame.setLayout(left_layout)
        left_frame.setStyleSheet("""
            QFrame {
                border: 3px solid;
                background-color: #333F50;
                border-radius: 9px; /* 圆角边框 */
            }
        """)

        center_frame = QFrame()
        center_frame.setLayout(center_layout)
        center_frame.setStyleSheet("""
            QFrame {
                border: 3px solid;
                background-color: #FFFFFF;
                border-radius: 9px; /* 圆角边框 */
            }
        """)

        right_frame = QFrame()
        right_frame.setLayout(right_layout)
        right_frame.setFixedWidth(320)
        right_frame.setStyleSheet("""
            QFrame {
                border: none;
                background-color: #FFFFFF;
                border-radius: 9px; /* 圆角边框 */
            }
        """)

        # 在分割器中添加 QFrame
        splitter.addWidget(left_frame)
        splitter.addWidget(center_frame)
        splitter.addWidget(right_frame)

        # 创建一个中心小部件，并设置布局
        self.setCentralWidget(splitter)


        # 初始化各列的内容
        self.create_left_column(left_layout)
        self.create_center_column(center_layout)
        self.create_right_column(right_layout)

    def create_left_column(self, layout):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # 设置滚动区域可调整大小
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #333F50;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #333F50;
                width: 10px;
                margin: 11px 0;
            }
            QScrollBar::handle:vertical {
                background-color: #888;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #555;
            }
            QScrollBar::add-line:vertical {
                border: none;
                background-color: #333F50;
                height: 10px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }
            QScrollBar::sub-line:vertical {
                border: none;
                background-color: #333F50;
                height: 10px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background-color: #333F50;
            }
            QScrollBar:horizontal {
                border: none;
                background-color: #333F50;
                height: 10px;
                margin: 0 11px;
            }
            QScrollBar::handle:horizontal {
                background-color: #888;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #555;
            }
            QScrollBar::add-line:horizontal {
                border: none;
                background-color: #333F50;
                width: 10px;
                subcontrol-position: right;
                subcontrol-origin: margin;
            }
            QScrollBar::sub-line:horizontal {
                border: none;
                background-color: #333F50;
                width: 10px;
                subcontrol-position: left;
                subcontrol-origin: margin;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background-color: #333F50;
            }
        """)

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_widget.setStyleSheet("""
            QWidget {
                border: none;
                background-color: #333F50;
            }
        """)

        select_excel_layout = QHBoxLayout()
        select_excel_button = QPushButton('请点击选择运价数据表格文件', self)
        select_excel_button.clicked.connect(self.selectExcelFile)
        select_excel_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
                font-size: 9pt; /* 字号 */
                background-color: #4A6A8C; /* 深色背景 */
                color: white; /* 文本颜色 */
                border: 1px solid #2E3D4F; /* 边框 */
                border-radius: 8px; /* 圆角边框 */
                padding: 12px; /* 填充 */
                margin: 5px; /* 外边距 */
                min-width: 200px; /* 最小宽度 */
                box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2); /* 轻微阴影 */
            }
            QPushButton:hover {
                background-color: #6A84A0; /* 悬停时的背景颜色 */
                color: #FFF; /* 悬停时文本颜色 */
                box-shadow: 3px 3px 9px rgba(0, 0, 0, 0.3); /* 悬停时的阴影 */
            }
            QPushButton:pressed {
                background-color: #3E506A; /* 按下时的背景颜色 */
                color: #FFF; /* 按下时文本颜色 */
                box-shadow: none; /* 按下时移除阴影 */
            }
            QPushButton:disabled {
                background-color: #888888; /* 禁用时的背景颜色 */
                color: #CCCCCC; /* 禁用时的文本颜色 */
                border: 1px solid #666666; /* 禁用时的边框 */
            }
        """)
        select_excel_layout.addWidget(select_excel_button)
        content_layout.addLayout(select_excel_layout)

        choose_sheet_layout = QHBoxLayout()
        # 创建标签
        choose_sheet_label = QLabel('请选择数据集：', self)
        choose_sheet_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        # 创建下拉菜单
        self.SheetComboBox = QComboBox(self)
        self.SheetComboBox.currentTextChanged.connect(self.updateDataset)
        self.SheetComboBox.setStyleSheet("""
            QComboBox {
                font-weight: bold;
                font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
                font-size: 9pt; /* 字号稍微调大 */
                background-color: #FFFFFF; /* 下拉框的背景颜色保持白色 */
                color: #333F50; /* 文本颜色保持与背景协调 */
                border: 2px solid #FFFFFF; /* 深蓝色边框，显眼 */
                border-radius: 9px; /* 圆角边框 */
                padding: 8px 9px; /* 内边距让内容不至于太紧凑 */
                min-width: 150px; /* 保证下拉框有足够的宽度 */
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px; /* 调整下拉按钮的宽度 */
                border-left-width: 1px;
                border-left-color: #4A90E2; /* 使用与边框一致的颜色 */
                border-left-style: solid;
                border-top-right-radius: 9px;
                border-bottom-right-radius: 9px;
                background-color: #4A6A8C; /* 下拉按钮的背景颜色 */
            }

            QComboBox QAbstractItemView {
                selection-background-color: #007BFF; /* 选中项背景颜色 */
                selection-color: #FFFFFF; /* 选中项文本颜色 */
                background-color: #F5F5F5; /* 下拉列表背景色为浅灰色 */
                border: 1px solid #4A90E2; /* 列表的边框颜色 */
                outline: none;
                padding: 5px;
            }

            QComboBox::item {
                padding: 8px; /* 每个下拉项的内边距 */
                background-color: transparent;
            }

            QComboBox::item:hover {
                background-color: #B4C7E7; /* 悬停时的背景颜色 */
                color: #333F50; /* 悬停时文本颜色 */
            }
        """)

        choose_sheet_layout.addWidget(choose_sheet_label)
        choose_sheet_layout.addWidget(self.SheetComboBox)
        content_layout.addLayout(choose_sheet_layout)

        choose_model_layout = QHBoxLayout()
        choose_model_label = QLabel('请选择预测模型：', self)
        choose_model_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        # 下拉菜单
        self.comboBox = QComboBox(self)
        for scrip_name in self.scripts:
            self.comboBox.addItem(scrip_name)
        self.comboBox.currentIndexChanged.connect(self.selectModel)
        self.comboBox.setStyleSheet("""
            QComboBox {
                font-weight: bold;
                font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
                font-size: 9pt; /* 字号稍微调大 */
                background-color: #FFFFFF; /* 下拉框的背景颜色保持白色 */
                color: #333F50; /* 文本颜色保持与背景协调 */
                border: 2px solid #FFFFFF; /* 深蓝色边框，显眼 */
                border-radius: 9px; /* 圆角边框 */
                padding: 8px 9px; /* 内边距让内容不至于太紧凑 */
                min-width: 150px; /* 保证下拉框有足够的宽度 */
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 30px; /* 调整下拉按钮的宽度 */
                border-left-width: 1px;
                border-left-color: #4A90E2; /* 使用与边框一致的颜色 */
                border-left-style: solid;
                border-top-right-radius: 9px;
                border-bottom-right-radius: 9px;
                background-color: #4A6A8C; /* 下拉按钮的背景颜色 */
            }

            QComboBox QAbstractItemView {
                selection-background-color: #007BFF; /* 选中项背景颜色 */
                selection-color: #FFFFFF; /* 选中项文本颜色 */
                background-color: #F5F5F5; /* 下拉列表背景色为浅灰色 */
                border: 1px solid #4A90E2; /* 列表的边框颜色 */
                outline: none;
                padding: 5px;
            }

            QComboBox::item {
                padding: 8px; /* 每个下拉项的内边距 */
                background-color: transparent;
            }

            QComboBox::item:hover {
                background-color: #B4C7E7; /* 悬停时的背景颜色 */
                color: #333F50; /* 悬停时文本颜色 */
            }
        """)

        choose_model_layout.addWidget(choose_model_label)
        choose_model_layout.addWidget(self.comboBox)
        content_layout.addLayout(choose_model_layout)

        label_layout = QHBoxLayout()
        self.label = QLabel('待坝需求：', self)
        self.label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        # 复选框
        self.checkBox_1 = QCheckBox('需要', self)
        self.checkBox_1.stateChanged.connect(self.checkBoxToggled)
        self.checkBox_1.setStyleSheet("""
            QCheckBox {
                spacing: 9px; /* 文本和框框之间的间距 */
                font-weight: bold; /* 加粗 */
                font-family: 'SimSun', serif; /* 使用宋体 */
                font-size: 9pt; /* 字号 */
                color: #FFFFFF; /* 文本颜色 */
                background-color: transparent; /* 透明背景色 */
                padding: 5px 9px; /* 内边距，增加视觉间隔 */
            }
            QCheckBox::indicator {
                width: 20px; /* 复选框宽度 */
                height: 20px; /* 复选框高度 */
                border: 2px solid #007BFF; /* 边框颜色 */
                border-radius: 5px; /* 圆角 */
                background-color: #F5F5F5; /* 背景颜色 */
            }
            QCheckBox::indicator:unchecked {
                background-color: #F5F5F5; /* 未选中时背景色 */
                border: 2px solid #007BFF; /* 未选中时边框颜色 */
            }
            QCheckBox::indicator:unchecked:hover {
                background-color: #E0E0E0; /* 悬停时背景色 */
            }
            QCheckBox::indicator:unchecked:pressed {
                background-color: #C0C0C0; /* 按下时背景色 */
            }
            QCheckBox::indicator:checked {
                background-color: #007BFF; /* 选中时背景色 */
                border: 2px solid #007BFF; /* 选中时边框颜色 */
                image: url(/path/to/checked-icon.png); /* 选中时的图标路径 */
            }
            QCheckBox::indicator:checked:hover {
                background-color: #0056B3; /* 选中且悬停时背景色 */
            }
            QCheckBox::indicator:checked:pressed {
                background-color: #00398C; /* 选中且按下时背景色 */
            }
        """)

        # 复选框
        self.checkBox_2 = QCheckBox('不需要', self)
        self.checkBox_2.stateChanged.connect(self.checkBoxToggled)
        self.checkBox_2.setStyleSheet("""
            QCheckBox {
                spacing: 9px; /* 文本和框框之间的间距 */
                font-weight: bold; /* 加粗 */
                font-family: 'SimSun', serif; /* 使用宋体 */
                font-size: 9pt; /* 字号 */
                color: #FFFFFF; /* 文本颜色 */
                background-color: transparent; /* 透明背景色 */
                padding: 5px 9px; /* 内边距，增加视觉间隔 */
            }
            QCheckBox::indicator {
                width: 20px; /* 复选框宽度 */
                height: 20px; /* 复选框高度 */
                border: 2px solid #007BFF; /* 边框颜色 */
                border-radius: 5px; /* 圆角 */
                background-color: #F5F5F5; /* 背景颜色 */
            }
            QCheckBox::indicator:unchecked {
                background-color: #F5F5F5; /* 未选中时背景色 */
                border: 2px solid #007BFF; /* 未选中时边框颜色 */
            }
            QCheckBox::indicator:unchecked:hover {
                background-color: #E0E0E0; /* 悬停时背景色 */
            }
            QCheckBox::indicator:unchecked:pressed {
                background-color: #C0C0C0; /* 按下时背景色 */
            }
            QCheckBox::indicator:checked {
                background-color: #007BFF; /* 选中时背景色 */
                border: 2px solid #007BFF; /* 选中时边框颜色 */
                image: url(/path/to/checked-icon.png); /* 选中时的图标路径 */
            }
            QCheckBox::indicator:checked:hover {
                background-color: #0056B3; /* 选中且悬停时背景色 */
            }
            QCheckBox::indicator:checked:pressed {
                background-color: #00398C; /* 选中且按下时背景色 */
            }
        """)

        self.label.hide()
        self.checkBox_1.hide()
        self.checkBox_2.hide()
        label_layout.addWidget(self.label)
        label_layout.addWidget(self.checkBox_1)
        label_layout.addWidget(self.checkBox_2)
        content_layout.addLayout(label_layout)

        price_index_layout = QHBoxLayout()
        self.price_index_label = QLabel('当前月份运价指数:', self)
        self.price_index_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.price_index_edit = QLineEdit(self)
        self.price_index_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.price_index_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.price_index_label.hide()
        self.price_index_edit.hide()
        price_index_layout.addWidget(self.price_index_label)
        price_index_layout.addWidget(self.price_index_edit)
        content_layout.addLayout(price_index_layout)

        gasoline_price_layout = QHBoxLayout()
        self.gasoline_price_label = QLabel('当前月份汽油价格:', self)
        self.gasoline_price_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.gasoline_price_edit = QLineEdit(self)
        self.gasoline_price_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.gasoline_price_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.gasoline_price_label.hide()
        self.gasoline_price_edit.hide()
        gasoline_price_layout.addWidget(self.gasoline_price_label)
        gasoline_price_layout.addWidget(self.gasoline_price_edit)
        content_layout.addLayout(gasoline_price_layout)

        diesel_price_layout = QHBoxLayout()
        self.diesel_price_label = QLabel('当前月份柴油价格:', self)
        self.diesel_price_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.diesel_price_edit = QLineEdit(self)
        self.diesel_price_edit.setFixedWidth(250)
        self.diesel_price_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.diesel_price_label.hide()
        self.diesel_price_edit.hide()
        diesel_price_layout.addWidget(self.diesel_price_label)
        diesel_price_layout.addWidget(self.diesel_price_edit)
        content_layout.addLayout(diesel_price_layout)

        water_level_layout = QHBoxLayout()
        self.water_level_label = QLabel('当前月份水位:', self)
        self.water_level_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.water_level_edit = QLineEdit(self)
        self.water_level_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.water_level_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.water_level_label.hide()
        self.water_level_edit.hide()
        water_level_layout.addWidget(self.water_level_label)
        water_level_layout.addWidget(self.water_level_edit)
        content_layout.addLayout(water_level_layout)

        water_transport_employee_layout = QHBoxLayout()
        self.water_transport_employee_label = QLabel('水上运输从业人数:', self)
        self.water_transport_employee_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.water_transport_employee_edit = QLineEdit(self)
        self.water_transport_employee_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.water_transport_employee_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.water_transport_employee_label.hide()
        self.water_transport_employee_edit.hide()
        water_transport_employee_layout.addWidget(self.water_transport_employee_label)
        water_transport_employee_layout.addWidget(self.water_transport_employee_edit)
        content_layout.addLayout(water_transport_employee_layout)

        tube_transport_employee_layout = QHBoxLayout()
        self.tube_transport_employee_label = QLabel('管道运输从业人数:', self)
        self.tube_transport_employee_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.tube_transport_employee_edit = QLineEdit(self)
        self.tube_transport_employee_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.tube_transport_employee_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.tube_transport_employee_label.hide()
        self.tube_transport_employee_edit.hide()
        tube_transport_employee_layout.addWidget(self.tube_transport_employee_label)
        tube_transport_employee_layout.addWidget(self.tube_transport_employee_edit)
        content_layout.addLayout(tube_transport_employee_layout)

        GDP_index_layout = QHBoxLayout()
        self.GDP_index_label = QLabel('GDP指数:', self)
        self.GDP_index_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.GDP_index_edit = QLineEdit(self)
        self.GDP_index_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.GDP_index_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.GDP_index_label.hide()
        self.GDP_index_edit.hide()
        GDP_index_layout.addWidget(self.GDP_index_label)
        GDP_index_layout.addWidget(self.GDP_index_edit)
        content_layout.addLayout(GDP_index_layout)

        CPI_index_layout = QHBoxLayout()
        self.CPI_index_label = QLabel('CPI指数:', self)
        self.CPI_index_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.CPI_index_edit = QLineEdit(self)
        self.CPI_index_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.CPI_index_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.CPI_index_label.hide()
        self.CPI_index_edit.hide()
        CPI_index_layout.addWidget(self.CPI_index_label)
        CPI_index_layout.addWidget(self.CPI_index_edit)
        content_layout.addLayout(CPI_index_layout)

        government_revenue_layout = QHBoxLayout()
        self.government_revenue_label = QLabel('国家财政收入:', self)
        self.government_revenue_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.government_revenue_edit = QLineEdit(self)
        self.government_revenue_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.government_revenue_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.government_revenue_label.hide()
        self.government_revenue_edit.hide()
        government_revenue_layout.addWidget(self.government_revenue_label)
        government_revenue_layout.addWidget(self.government_revenue_edit)
        content_layout.addLayout(government_revenue_layout)

        electricity_consumption_layout = QHBoxLayout()
        self.electricity_consumption_label = QLabel('电能消费总量:', self)
        self.electricity_consumption_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.electricity_consumption_edit = QLineEdit(self)
        self.electricity_consumption_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.electricity_consumption_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.electricity_consumption_label.hide()
        self.electricity_consumption_edit.hide()
        electricity_consumption_layout.addWidget(self.electricity_consumption_label)
        electricity_consumption_layout.addWidget(self.electricity_consumption_edit)
        content_layout.addLayout(electricity_consumption_layout)

        oil_consumption_layout = QHBoxLayout()
        self.oil_consumption_label = QLabel('石油消费总量:', self)
        self.oil_consumption_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.oil_consumption_edit = QLineEdit(self)
        self.oil_consumption_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.oil_consumption_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.oil_consumption_label.hide()
        self.oil_consumption_edit.hide()
        oil_consumption_layout.addWidget(self.oil_consumption_label)
        oil_consumption_layout.addWidget(self.oil_consumption_edit)
        content_layout.addLayout(oil_consumption_layout)

        port_throughput_layout = QHBoxLayout()
        self.port_throughput_label = QLabel('成品油港口吞吐量:', self)
        self.port_throughput_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.port_throughput_edit = QLineEdit(self)
        self.port_throughput_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.port_throughput_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.port_throughput_label.hide()
        self.port_throughput_edit.hide()
        port_throughput_layout.addWidget(self.port_throughput_label)
        port_throughput_layout.addWidget(self.port_throughput_edit)
        content_layout.addLayout(port_throughput_layout)

        refined_oil_consumption_layout = QHBoxLayout()
        self.refined_oil_consumption_label = QLabel('成品油港口吞吐量:', self)
        self.refined_oil_consumption_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.refined_oil_consumption_edit = QLineEdit(self)
        self.refined_oil_consumption_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.refined_oil_consumption_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.refined_oil_consumption_label.hide()
        self.refined_oil_consumption_edit.hide()
        refined_oil_consumption_layout.addWidget(self.refined_oil_consumption_label)
        refined_oil_consumption_layout.addWidget(self.refined_oil_consumption_edit)
        content_layout.addLayout(refined_oil_consumption_layout)

        capacity_inventory_layout = QHBoxLayout()
        self.capacity_inventory_label = QLabel('运力保有量:', self)
        self.capacity_inventory_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.capacity_inventory_edit = QLineEdit(self)
        self.capacity_inventory_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.capacity_inventory_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.capacity_inventory_label.hide()
        self.capacity_inventory_edit.hide()
        capacity_inventory_layout.addWidget(self.capacity_inventory_label)
        capacity_inventory_layout.addWidget(self.capacity_inventory_edit)
        content_layout.addLayout(capacity_inventory_layout)

        tube_transportation_scale_layout = QHBoxLayout()
        self.tube_transportation_scale_label = QLabel('管道运输规模:', self)
        self.tube_transportation_scale_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.tube_transportation_scale_edit = QLineEdit(self)
        self.tube_transportation_scale_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.tube_transportation_scale_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        self.tube_transportation_scale_label.hide()
        self.tube_transportation_scale_edit.hide()
        tube_transportation_scale_layout.addWidget(self.tube_transportation_scale_label)
        tube_transportation_scale_layout.addWidget(self.tube_transportation_scale_edit)
        content_layout.addLayout(tube_transportation_scale_layout)

        forecast_steps_layout = QHBoxLayout()
        forecast_steps_label = QLabel('需要预测的步长:', self)
        forecast_steps_label.setStyleSheet("""
            font-family: 'SimSun', sans-serif; /* 使用微软雅黑字体 */
            font-size: 9pt; /* 稍微增加字号 */
            text-align: center; /* 居中对齐 */
            font-weight: bold; /* 加粗 */
            color: #FFFFFF; /* 使用白色字体以在深色背景上有良好对比度 */
            padding: 5px 9px; /* 增加内边距 */
            margin: 5px 0; /* 增加外边距使其与其他组件分隔 */
            border: none; /* 移除边框 */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5); /* 增加轻微文本阴影 */
        """)
        self.forecast_number_edit = QLineEdit(self)
        self.forecast_number_edit.setFixedWidth(250)  # 设置固定宽度为250像素
        self.forecast_number_edit.setStyleSheet("""
            QLineEdit {
                font-family: 'SimSun', serif; /* 使用宋体字体 */
                font-size: 9pt; /* 字号 */
                text-align: left; /* 文本左对齐 */
                font-weight: bold; /* 加粗文本 */
                background-color: #FFFFFF; /* 输入框背景色 */
                border: 1px solid #CCCCCC; /* 边框颜色 */
                border-radius: 5px; /* 边框圆角 */
                padding: 9px; /* 内边距，增加内容与边界的距离 */
                color: #333333; /* 文本颜色 */
                min-height: 35px; /* 输入框最小高度，增强可点击性 */
            }
            QLineEdit:focus {
                border: 1px solid #4A90E2; /* 输入框获得焦点时的边框颜色 */
                background-color: #F0F8FF; /* 焦点时的背景色，柔和的蓝色 */
                outline: none; /* 移除默认外框 */
            }
            QLineEdit:disabled {
                background-color: #EFEFEF; /* 输入框禁用时的背景色 */
                color: #A9A9A9; /* 禁用时的文本颜色 */
                border: 1px solid #D5D5D5; /* 禁用时的边框颜色 */
            }
        """)
        forecast_steps_layout.addWidget(forecast_steps_label)
        forecast_steps_layout.addWidget(self.forecast_number_edit)
        content_layout.addLayout(forecast_steps_layout)

        # 按钮
        button_layout = QHBoxLayout()
        button = QPushButton('开始预测', self)
        button.clicked.connect(self.buttonClicked)
        button.setStyleSheet("""
            QPushButton {
                font-family: 'SimSun', sans-serif; /* 使用宋体字体 */
                font-size: 9pt; /* 增大字号 */
                text-align: center; 
                font-weight: bold;
                background-color: #C55A11; /* 按钮背景颜色 */
                color: white; /* 文本颜色 */
                border: none; /* 无边框 */
                border-radius: 8px; /* 增加圆角 */
                padding: 12px; /* 增加内边距，提高可点击区域 */
                min-width: 150px; /* 设置最小宽度，使按钮看起来更大 */
            }
            QPushButton:hover {
                background-color: #0056b3; /* 悬停时的背景颜色 */
            }
            QPushButton:pressed {
                background-color: #003d80; /* 按下时的背景颜色 */
            }
        """)
        button_layout.addWidget(button)
        content_layout.addLayout(button_layout)

        content_layout.setAlignment(Qt.AlignTop)
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

    def selectExcelFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择Excel文件",
            "",
            "Excel Files (*.xlsx *.xls);;All Files (*)",
            options=options
        )
        if self.file_path:
            # 获取Excel文件的所有Sheet名称
            sheet_names = self.getSheetNames(self.file_path)
            self.SheetComboBox.clear()  # 清空下拉菜单
            self.SheetComboBox.addItems(sheet_names)  # 添加Sheet名称到下拉菜单
            self.result_text_edit.append(f"选择了文件: {self.file_path}")
            self.result_text_edit.append(f"可选数据集有:\n{','.join(sheet_names)}")
            self.result_text_edit.append('-' * 50)
            self.result_text_edit.append(f"当前选择数据集为：{self.SheetComboBox.currentText()}")
            self.result_text_edit.append('-' * 50)

    def updateDataset(self, sheet_name):
        self.result_text_edit.append(f"当前选择数据集为： {sheet_name}")

    def getSheetNames(self, file_name):
        # 使用pandas读取Excel文件并获取所有Sheet名称
        xls = pd.ExcelFile(file_name)
        return xls.sheet_names

    def create_center_column(self, layout):
        vertical_splitter = QSplitter(Qt.Vertical)
        vertical_splitter.setStyleSheet("""
            QSplitter {
                border: none;
                background-color: white; /* 设置背景颜色 */
            }
        """)
        # 图像显示区域
        self.imageLabel = QLabel(self)
        self.imageLabel.setFixedSize(self.img_width, self.img_height)
        self.imageLabel.setStyleSheet("border:none;")
        # 初始化默认图片
        self.loadImage()
        self.result_text_edit = QTextEdit(self)
        self.result_text_edit.setReadOnly(True)
        self.result_text_edit.setStyleSheet("""
            QTextEdit {
                border: 1px solid #CCCCCC; /* 边框颜色为浅灰色 */
                border-radius: 4px; /* 边框圆角 */
                background-color: #FFFFFF; /* 背景色为白色 */
            }
            QTextEdit QScrollBar:vertical {
                border: none;
                background-color: rgba(0, 0, 0, 0%);
                width: 10px;
                margin: 11px 0;
            }
            QTextEdit QScrollBar::handle:vertical {
                background-color: #888;
                min-height: 20px;
                border-radius: 5px;
            }
            QTextEdit QScrollBar::handle:vertical:hover {
                background-color: #555;
            }
            QTextEdit QScrollBar::add-line:vertical {
                border: none;
                background-color: rgba(0, 0, 0, 0%);
                height: 10px;
                subcontrol-position: bottom;
                subcontrol-origin: margin;
            }
            QTextEdit QScrollBar::sub-line:vertical {
                border: none;
                background-color: rgba(0, 0, 0, 0%);
                height: 10px;
                subcontrol-position: top;
                subcontrol-origin: margin;
            }
            QTextEdit QScrollBar::add-page:vertical, QTextEdit QScrollBar::sub-page:vertical {
                background-color: rgba(0, 0, 0, 0%);
            }
            QTextEdit QScrollBar:horizontal {
                border: none;
                background-color: rgba(0, 0, 0, 0%);
                height: 10px;
                margin: 0 11px;
            }
            QTextEdit QScrollBar::handle:horizontal {
                background-color: #888;
                min-width: 20px;
                border-radius: 5px;
            }
            QTextEdit QScrollBar::handle:horizontal:hover {
                background-color: #555;
            }
            QTextEdit QScrollBar::add-line:horizontal {
                border: none;
                background-color: rgba(0, 0, 0, 0%);
                width: 10px;
                subcontrol-position: right;
                subcontrol-origin: margin;
            }
            QTextEdit QScrollBar::sub-line:horizontal {
                border: none;
                background-color: rgba(0, 0, 0, 0%);
                width: 10px;
                subcontrol-position: left;
                subcontrol-origin: margin;
            }
            QTextEdit QScrollBar::add-page:horizontal, QTextEdit QScrollBar::sub-page:horizontal {
                background-color: rgba(0, 0, 0, 0%);
            }
        """)
        vertical_splitter.addWidget(self.imageLabel)
        vertical_splitter.addWidget(self.result_text_edit)
        layout.addWidget(vertical_splitter)

    def create_right_column(self, layout):
        # 数据显示窗口
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(2)
        # 设置固定的列宽
        self.tableWidget.setColumnWidth(0, 122)  # 设置第一列宽度为100像素
        self.tableWidget.setColumnWidth(1, 122)  # 设置第二列宽度为90像素
        # 隐藏内置的水平表头
        self.tableWidget.horizontalHeader().setVisible(False)
        # 隐藏内置的垂直表头
        # self.tableWidget.verticalHeader().setVisible(False)
        # 初始化行数，包括表头行和数据行
        self.tableWidget.setRowCount(2)  # 表头行 + 数据行
        # 在第一行合并单元格
        self.tableWidget.setItem(0, 0, QTableWidgetItem('价格预测数据'))
        self.tableWidget.setSpan(0, 0, 1, 2)  # 合并第一行的两个单元格
        # 设置第二行的列头
        self.tableWidget.setItem(1, 0, QTableWidgetItem('时间'))
        self.tableWidget.setItem(1, 1, QTableWidgetItem('价格'))
        self.tableWidget.item(0, 0).setTextAlignment(Qt.AlignCenter)  # 居中文本
        self.tableWidget.item(1, 0).setTextAlignment(Qt.AlignCenter)  # 居中文本
        self.tableWidget.item(1, 1).setTextAlignment(Qt.AlignCenter)  # 居中文本
        # 设置表格的样式表
        self.tableWidget.setStyleSheet("""
            QTableWidget {
                font-weight: bold;
                font-family: 'FangSong', sans-serif; /* 字体 */
                font-size: 9pt;       /* 字号 */
                color: #333;           /* 文字颜色 */
                background-color: #F8F8F8; /* 背景颜色 */
                gridline-color: #ccc;  /* 网格线颜色 */
                show-grid: 1;          /* 显示网格线 */
                alternate-background-color: #ECECEC; /* 交替行颜色 */
            }
            QTableWidget::item {
                padding: 5px;          /* 内边距 */
                border: 1px solid #ddd; /* 边框 */
                text-align: center;    /* 内容居中 */
            }
            QTableWidget::item:selected {
                background-color: #8EA9DB; /* 选中行的背景色 */
            }
        """)
        layout.addWidget(self.tableWidget)

    def displayDataFrameInTable(self, df):
        # 清空表格
        # self.tableWidget.clearContents()
        self.tableWidget.setRowCount(df.shape[0] + 2)  # 设置行数

        # 填充表格
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[row, col]))
                self.tableWidget.setItem(row + 2, col, item)

    def buttonClicked(self):
        print("Button clicked!")
        try:
            print("Button clicked!")
            if not self.file_path:
                raise ValueError('预测失败！未选择excel表格！')
            if not self.SheetComboBox.currentText():
                raise ValueError('预测失败！未选择数据集！')
            if not self.forecast_number_edit.text():
                raise ValueError('预测失败！未输入预测步数！')
            if not self.comboBox.currentText():
                raise ValueError('预测失败！未选择预测模型！')

            # 如果所有检查都通过，则继续执行
            if self.process is not None and self.process.state() != QProcess.NotRunning:
                self.process.terminate()
                self.process.waitForFinished(-1, Qt.CoarseTimer)

            selected_model = self.comboBox.currentText()
            script_path = self.scripts[selected_model]
            forecast_steps = self.forecast_number_edit.text()
            sheet_name = self.SheetComboBox.currentText()
            file_path = self.file_path
            self.predict_data_path = self.predict_data[selected_model]
            self.show_img_path = self.img_path[selected_model]

            print(self.predict_data_path)
            print(forecast_steps, type(forecast_steps))

            # 使用 QProcess 调用 EMD-ARIMA.py 脚本
            self.process = QProcess(self)
            self.process.readyReadStandardOutput.connect(self.handleStdOutput)
            self.process.readyReadStandardError.connect(self.handleStdError)
            self.process.stateChanged.connect(self.handleStateChanged)
            self.process.start("python", [script_path, file_path, sheet_name, forecast_steps])

        except ValueError as e:
            self.result_text_edit.append(str(e))

    def closeEvent(self, event):
        if self.process is not None:
            self.process.terminate()
            self.process.waitForFinished()
        event.accept()

    def handleStdOutput(self):
        data = self.process.readAllStandardOutput().data().decode()
        cleaned_data = re.sub(r'\x1b\[[0-9;]*m', '', data)
        self.result_text_edit.append(cleaned_data.strip())

    def handleStdError(self):
        data = self.process.readAllStandardError().data().decode()
        # 移除 ANSI 转义序列
        cleaned_data = re.sub(r'\x1b\[[0-9;]*m', '', data)
        self.result_text_edit.append(cleaned_data.strip())

    def handleStateChanged(self, state):
        states = {
            QProcess.NotRunning: 'Finish running',
            QProcess.Starting: 'Starting',
            QProcess.Running: 'Start running',
        }
        state_name = states[state]
        if state_name == 'Finish running':
            self.loadImage(self.show_img_path)
            self.displayDataFrameInTable(pd.read_csv(self.predict_data_path))
        self.result_text_edit.append(f"Process state changed: {state_name}")

    def selectModel(self, index):
        if index in [2, 3]:
            self.label.show()
            self.checkBox_1.show()
            self.checkBox_2.show()
            self.price_index_label.show()
            self.price_index_edit.show()
            self.gasoline_price_label.show()
            self.gasoline_price_edit.show()
            self.diesel_price_label.show()
            self.diesel_price_edit.show()
            self.water_level_label.show()
            self.water_level_edit.show()
            self.water_transport_employee_label.show()
            self.water_transport_employee_edit.show()
            self.tube_transport_employee_label.show()
            self.tube_transport_employee_edit.show()
            self.GDP_index_label.show()
            self.GDP_index_edit.show()
            self.CPI_index_label.show()
            self.CPI_index_edit.show()
            self.government_revenue_label.show()
            self.government_revenue_edit.show()
            self.electricity_consumption_label.show()
            self.electricity_consumption_edit.show()
            self.oil_consumption_label.show()
            self.oil_consumption_edit.show()
            self.port_throughput_label.show()
            self.port_throughput_edit.show()
            self.refined_oil_consumption_label.show()
            self.refined_oil_consumption_edit.show()
            self.capacity_inventory_label.show()
            self.capacity_inventory_edit.show()
            self.tube_transportation_scale_label.show()
            self.tube_transportation_scale_edit.show()
        else:
            self.label.hide()
            self.checkBox_1.hide()
            self.checkBox_2.hide()
            self.price_index_label.hide()
            self.price_index_edit.hide()
            self.gasoline_price_label.hide()
            self.gasoline_price_edit.hide()
            self.diesel_price_label.hide()
            self.diesel_price_edit.hide()
            self.water_level_label.hide()
            self.water_level_edit.hide()
            self.water_transport_employee_label.hide()
            self.water_transport_employee_edit.hide()
            self.tube_transport_employee_label.hide()
            self.tube_transport_employee_edit.hide()
            self.GDP_index_label.hide()
            self.GDP_index_edit.hide()
            self.CPI_index_label.hide()
            self.CPI_index_edit.hide()
            self.government_revenue_label.hide()
            self.government_revenue_edit.hide()
            self.electricity_consumption_label.hide()
            self.electricity_consumption_edit.hide()
            self.oil_consumption_label.hide()
            self.oil_consumption_edit.hide()
            self.port_throughput_label.hide()
            self.port_throughput_edit.hide()
            self.refined_oil_consumption_label.hide()
            self.refined_oil_consumption_edit.hide()
            self.capacity_inventory_label.hide()
            self.capacity_inventory_edit.hide()
            self.tube_transportation_scale_label.hide()
            self.tube_transportation_scale_edit.hide()
        self.result_text_edit.append(f"Selected Model: {self.model_name[index]}")
        print(f"Selected option: {self.model_name[index]}")

    def checkBoxToggled(self, state):
        if state == Qt.Checked:
            print("Checkbox checked")
        else:
            print("Checkbox unchecked")

    def selectBoxChanged(self, index):
        print(f"Selected item: {index}")

    def loadImage(self, image_path=None):
        # 如果用户没有选择图片路径，使用默认图片路径
        if not image_path:
            image_path = self.default_image_path
        print(image_path)
        # 使用scaledToHeight方法按高度缩放图像，同时保持宽高比
        pixmap = QPixmap(image_path).scaled(self.img_width, self.img_height, Qt.KeepAspectRatio,
                                            Qt.SmoothTransformation)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setAlignment(Qt.AlignCenter)


def create_database():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    # 如果用户表不存在，则创建用户表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Login')
        # 设置窗口图标
        self.setWindowIcon(QIcon('./fig/logo_HD.jpg'))

        # 创建布局
        layout = QVBoxLayout()

        # 创建用户名和密码输入框
        self.username_label = QLabel('Username:')
        self.username_input = QLineEdit(self)

        self.password_label = QLabel('Password:')
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)  # 隐藏密码

        # 创建按钮
        self.login_button = QPushButton('Login')
        self.register_button = QPushButton('Register')

        # 连接按钮点击事件
        self.login_button.clicked.connect(self.login)
        self.register_button.clicked.connect(self.open_register)

        # 添加控件到布局
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.login_button)
        button_layout.addWidget(self.register_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        # 验证用户名和密码
        if self.validate_login(username, password):
            QMessageBox.information(self, 'Success', 'Login successful!')
            self.accept()
        else:
            QMessageBox.warning(self, 'Error', 'Incorrect username or password')

    def validate_login(self, username, password):
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = cursor.fetchone()
        conn.close()
        return user is not None

    def open_register(self):
        register_dialog = RegisterDialog()
        register_dialog.exec_()


class RegisterDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('User Register')
        # 设置窗口图标
        self.setWindowIcon(QIcon('./fig/logo_HD.jpg'))

        # 创建布局
        layout = QVBoxLayout()

        # 创建输入框
        self.username_label = QLabel('Username:')
        self.username_input = QLineEdit(self)

        self.password_label = QLabel('Password:')
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)

        self.code_label = QLabel('Registration Code:')
        self.code_input = QLineEdit(self)

        # 创建按钮
        self.register_button = QPushButton('Register')

        # 连接按钮点击事件
        self.register_button.clicked.connect(self.register)

        # 添加控件到布局
        layout.addWidget(self.username_label)
        layout.addWidget(self.username_input)
        layout.addWidget(self.password_label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.code_label)
        layout.addWidget(self.code_input)
        layout.addWidget(self.register_button)
        self.setLayout(layout)

    def register(self):
        username = self.username_input.text()
        password = self.password_input.text()
        registration_code = self.code_input.text()

        # 校验注册码
        if registration_code != 'bO61ZU':
            QMessageBox.warning(self, 'Error', 'Invalid registration code!')
            return

        # 插入新用户到数据库
        if self.add_user(username, password):
            QMessageBox.information(self, 'Success', 'Registration successful!')
            self.accept()
        else:
            QMessageBox.warning(self, 'Error', 'Username already exists')

    def add_user(self, username, password):
        try:
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./fig/logo.ico'))
    # 创建数据库并初始化表
    create_database()

    login_dialog = LoginDialog()
    if login_dialog.exec_() == QDialog.Accepted:
        window = MainWindow()
        window.showMaximized()

    # window = MainWindow()
    # window.showMaximized()

    sys.exit(app.exec_())
