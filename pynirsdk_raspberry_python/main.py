import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,QWidget, QPushButton, QLabel, QTextEdit, QFrame, QLineEdit, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIntValidator
# import data   # 导入数据采集模块， 本机测试的时候记得注释掉， 如果在嵌入式设备上测试就取消注释
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import font_manager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
from predictor import SpectrumPredictor
import pandas as pd
import logging

# 字体这里出现了问题，但是不影响正常使用，所以这里把字体报错屏蔽掉了，等后续会把字体部分修好就可以了
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ---------- 全局 QSS 样式表 ----------
# 这部分是ai写的直接给ai改就行， 这部分是界面美化的代码，不参加主功能的实现， 更改代码的时候不要更改这个变量名， 直接修改里面的样式就行了
STYLE_SHEET = """
QMainWindow {
    background-color: #f5f6fa;
}

QFrame#card {
    background-color: white;
    border-radius: 8px;
    border: none;
}

QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #4a90e2, stop:1 #357abd);
    color: white;
    border: none;
    border-radius: 5px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: bold;
    min-width: 80px;
}

QPushButton:hover {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #5a9ef2, stop:1 #458acd);
}

QPushButton:pressed {
    background-color: #2c5a8c;
}

QPushButton#cancelButton {
    background-color: #e0e0e0;
    color: #333;
}

QPushButton#cancelButton:hover {
    background-color: #d0d0d0;
}

QLabel {
    color: #2c3e50;
    font-size: 14px;
    font-weight: 500;  /* 半粗体 */
}

QLabel#titleLabel {
    font-size: 18px;
    font-weight: bold;
    color: #2c3e50;
    padding: 10px;
}

QTextEdit {
    background-color: white;
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 8px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
}

QStatusBar {
    background-color: #e0e4e8;
    color: #2c3e50;
}

QLineEdit {
    background-color: white;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 14px;
    color: #2c3e50;
    selection-background-color: #4a90e2;
    selection-color: white;
}
QLineEdit:hover {
    border-color: #b6c2cc;
}
QLineEdit:focus {
    border-color: #4a90e2;
    outline: none;
    box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.2);
}
QLineEdit:disabled {
    background-color: #f5f6fa;
    color: #a0aec0;
}

QFrame#inputCard {
    background-color: #f8fafc;
    border-radius: 6px;
    padding: 8px;
}

"""
#---------- 全局 QSS 样式表 ----------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIR光谱分析系统")
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()

        self.predictor = self.load_model()
        self.startButton.clicked.connect(self.start_collection)
        self.cancalButton.clicked.connect(self.on_change_clicked)
        self.restartButton.clicked.connect(self.restart_clicked)
        # 新增：连接“再次重复”按钮
        self.repeatButton.clicked.connect(self.repeat_again)
        self.cancalRepeatButton.clicked.connect(self.cancel_repeat_mode)

        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_collect)
        self.collect_count = 0
        self.max_collect = 10
        self.high_conf_records = []
        self.last_result = None

        self.canvas = None


    def initUI(self):
        """界面初始化，三段式布局"""
        cerntral = self.centralWidget()
        if cerntral is None:
            cerntral = QWidget()
            self.setCentralWidget(cerntral)

        main_layout = QHBoxLayout(cerntral)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ---------- 左侧面板 ----------
        left_panel = QFrame()
        left_panel.setObjectName("card")
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        title = QLabel("控制面板")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        self.startButton = QPushButton("▶ 开始采集")
        self.startButton.setMinimumHeight(40)
        left_layout.addWidget(self.startButton)

        self.restartButton = QPushButton("▶ 重复采集模式")
        self.restartButton.setMinimumHeight(40)
        left_layout.addWidget(self.restartButton)

        # 替换原有的两个 HBoxLayout 为以下网格布局
        grid_layout = QGridLayout()
        grid_layout.setVerticalSpacing(10)   # 行间距
        grid_layout.setColumnMinimumWidth(0, 80)  # 标签列宽
        grid_layout.setColumnMinimumWidth(1, 100) # 输入框列宽

        grid_layout.addWidget(QLabel("重复次数:"), 0, 0, Qt.AlignRight)
        self.count_edit = QLineEdit("10")
        self.count_edit.setValidator(QIntValidator(1, 999, self))
        grid_layout.addWidget(self.count_edit, 0, 1)

        grid_layout.addWidget(QLabel("间隔(秒):"), 1, 0, Qt.AlignRight)
        self.interval_edit = QLineEdit("4")
        self.interval_edit.setValidator(QIntValidator(1, 60, self))
        grid_layout.addWidget(self.interval_edit, 1, 1)

        left_layout.addLayout(grid_layout)

        self.cancalButton = QPushButton("✖ 清空")
        self.cancalButton.setObjectName("cancelButton")
        self.cancalButton.setMinimumHeight(40)
        left_layout.addWidget(self.cancalButton)

        left_layout.addStretch()
        status_frame = QFrame()
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(10, 10, 10, 10)

        self.model_status_label = QLabel("模型状态:")
        self.model_status_label.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(self.model_status_label)

        self.model_detail_label = QLabel("")
        self.model_detail_label.setWordWrap(True)
        status_layout.addWidget(self.model_detail_label)

        left_layout.addWidget(status_frame)

        # ---------- 中间面板（光谱图） ----------
        mid_panel = QFrame()
        mid_panel.setObjectName("card")
        mid_layout = QVBoxLayout(mid_panel)

        plot_title = QLabel("实时光谱图")
        plot_title.setObjectName("titleLabel")
        mid_layout.addWidget(plot_title)

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.addWidget(self.plot_container)

        # ---------- 右侧面板 ----------
        right_panel = QFrame()
        right_panel.setObjectName("card")
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)

        result_title = QLabel("识别结果")
        result_title.setObjectName("titleLabel")
        right_layout.addWidget(result_title)

        # 创建“再次重复”和“取消”按钮，放入水平框架
        self.repeatButton = QPushButton("再次重复")
        self.repeatButton.setMinimumHeight(40)
        self.cancalRepeatButton = QPushButton("取消")
        self.cancalRepeatButton.setMinimumHeight(40)
        self.cancalRepeatButton.setObjectName("cancelButton")  # 沿用取消样式

        self.repeat_Button_frame = QWidget()
        button_layout = QHBoxLayout(self.repeat_Button_frame)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(self.repeatButton)
        button_layout.addWidget(self.cancalRepeatButton)
        self.repeat_Button_frame.setVisible(False)  # 初始隐藏
        right_layout.addWidget(self.repeat_Button_frame)

        # 结果显示卡片
        result_card = QFrame()
        result_card.setStyleSheet("background-color: #f0f3f8; border-radius: 5px; padding: 10px;")
        result_card_layout = QVBoxLayout(result_card)

        self.result_label = QLabel("等待采集...")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        self.result_label.setAlignment(Qt.AlignCenter)
        result_card_layout.addWidget(self.result_label)
        right_layout.addWidget(result_card)

        # 概率标签和文本框
        prob_label = QLabel("各类别概率")
        prob_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        right_layout.addWidget(prob_label)

        self.prob_text = QTextEdit()
        self.prob_text.setMaximumHeight(300)
        right_layout.addWidget(self.prob_text)

        right_layout.addStretch()

        # 组合主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(mid_panel, 1)
        main_layout.addWidget(right_panel)


    def load_model(self):
        """加载训练好的模型"""

        try:
            results_dir = os.path.join(os.path.dirname(__file__), "data")
            if not os.path.exists(results_dir):
                print("未找到实验结果文件夹")
                return None
            exp_folders = [f for f in os.listdir(results_dir) if f.startswith("experiment_")]
            if not exp_folders:
                print("未找到实验文件夹")
                return None
            latest_exp = sorted(exp_folders)[-1]
            model_path = os.path.join(results_dir, latest_exp, "best_model.pth")
            if not os.path.exists(model_path):
                print(f"模型文件不存在: {model_path}")
                return None
            from predictor import SpectrumPredictor
            predictor = SpectrumPredictor(model_path)
            print("模型加载成功！")
            return predictor
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None


    def start_collection(self):
        print("start clicked")
        if not self.predictor:
            self.result_label.setText("模型未加载，无法进行预测")
            self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")
            return
        spectrum_data = self.collect_spectrum()
        if spectrum_data['success']:
            print("数据采集成功，波长和强度已更新")
            self.draw_spectrum(spectrum_data['wavelengths'], spectrum_data['intensities'])
            self.recognize_spectrum(spectrum_data)
        else:
            print("数据采集失败:", spectrum_data['message'])
            self.result_label.setText(f"采集失败: {spectrum_data['message']}")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")


    def restart_clicked(self):
        """重复采集模式：根据用户设置的次数和间隔进行多次采集"""
        print("重复模式开始")
        self.start_repeat_from_input()

    def timer_collect(self):
        """定时器槽函数"""
        if self.collect_count >= self.max_collect:
            self.timer.stop()
            self.display_ten_results()
            self.result_label.setText("重复运行结束")
            self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: blue;")
            self.show_repeat_buttons()
            return

        print(f"第 {self.collect_count+1} 次采集")
        spectrum_data = self.collect_spectrum()
        if spectrum_data['success']:
            self.draw_spectrum(spectrum_data['wavelengths'], spectrum_data['intensities'])
            self.recognize_spectrum(spectrum_data)
            if self.last_result and self.last_result.get('success'):
                label = self.last_result.get('predicted_label')
                conf = self.last_result.get('confidence')
                if label is not None and conf is not None:
                    self.high_conf_records.append({'label': label, 'confidence': conf})
        else:
            print(f"第 {self.collect_count+1} 次采集失败: {spectrum_data['message']}")
            self.high_conf_records.append({'label': '采集失败', 'confidence': 0.0})

        self.collect_count += 1
        if self.collect_count >= self.max_collect:
            self.timer.stop()
            self.display_ten_results()
            self.result_label.setText("重复运行结束")
            self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: blue;")
            self.show_repeat_buttons()

    def show_repeat_buttons(self):
        """隐藏左面板原按钮，显示右侧按钮框架"""
        self.startButton.setVisible(False)
        self.restartButton.setVisible(False)
        self.cancalButton.setVisible(False)
        self.repeat_Button_frame.setVisible(True)

    def hide_repeat_buttons(self):
        """隐藏右侧按钮框架，显示左面板原按钮"""
        self.startButton.setVisible(True)
        self.restartButton.setVisible(True)
        self.cancalButton.setVisible(True)
        self.repeat_Button_frame.setVisible(False)

    def repeat_again(self):
        """再次重复：重新开始十次采集（使用当前输入框的值）"""
        print("再次重复")
        # 先尝试启动，成功后再切换按钮
        if self.start_repeat_from_input():
            self.hide_repeat_buttons()  # 隐藏右侧按钮框架，显示左面板原按钮

    def cancel_repeat_mode(self):
        """取消重复模式：恢复原按钮，清空记录和光谱图"""
        print("取消重复模式")
        self.timer.stop()
        self.hide_repeat_buttons()
        self.high_conf_records.clear()
        self.prob_text.clear()
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
        self.result_label.setText("等待采集...")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")

    def display_ten_results(self):
        """在概率显示区域显示十次的历史记录"""
        if not self.high_conf_records:
            self.prob_text.setPlainText("无有效记录")
            return
        lines = []
        for i, record in enumerate(self.high_conf_records, 1):
            label = record['label']
            conf = record['confidence']
            lines.append(f"{i}. {label}: {conf:.2%}")
        text = "\n".join(lines)
        self.prob_text.setPlainText(text)

    def collect_spectrum(self):
        """采集数据（测试模式从Excel随机选取）"""
        print("正在采集数据...")
        # 实际硬件采集请取消注释以下代码块
        # try:
        #     import data
        #     return data.acquire_and_plot_spectrum()
        # except Exception as e:
        #     return {'success': False, 'wavelengths': [], 'intensities': [], 'message': str(e)}

        # 测试模式
        print("测试模式：从 test/试验数据2.xlsx 随机选取一个样本")
        try:
            file_path = os.path.join(os.path.dirname(__file__), "test", "试验数据2.xlsx")
            df = pd.read_excel(file_path, header=None)
            wavelengths = df.iloc[0, :-1].tolist()
            data_rows = df.iloc[1:, :-1].values.astype(np.float32)
            random_idx = np.random.randint(0, len(data_rows))
            intensities = data_rows[random_idx].tolist()
            print(f"成功从文件第 {random_idx+2} 行选取样本")
            return {
                'success': True,
                'wavelengths': wavelengths,
                'intensities': intensities,
                'message': f'从文件随机选取第 {random_idx+2} 行作为测试数据'
            }
        except Exception as e:
            print(f"读取文件失败: {e}")
            wavelengths = list(range(400, 628))
            intensities = list(np.random.randn(228) * 100 + 500)
            return {
                'success': True,
                'wavelengths': wavelengths,
                'intensities': intensities,
                'message': '文件读取失败，使用随机生成数据'
            }

    def recognize_spectrum(self, spectrum_data):
        """识别光谱并显示结果，保存预测结果到 self.last_result"""
        try:
            result = self.predictor.predict(
                spectrum_data['wavelengths'],
                spectrum_data['intensities']
            )
            self.last_result = result  # 保存本次结果供记录使用
            print("预测结果已获取，准备更新显示")
            self.update_result_display(result)
            print("update_result_display 调用完成")
        except Exception as e:
            self.result_label.setText(f"识别失败: {str(e)}")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
            self.last_result = None

    def on_change_clicked(self):
        """清空按钮：清除历史记录、概率文本和光谱图"""
        self.high_conf_records.clear()
        self.prob_text.clear()
        self.result_label.clear()
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None
        print("清空")

    def draw_spectrum(self, wavelengths, intensities):
        """绘制光谱图"""
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()

        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        ax.plot(wavelengths, intensities, color='#4a90e2', linewidth=2)
        ax.set_xlabel("Wavelength (nm)", fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.set_title("Real-time Spectrum", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fc')
        fig.patch.set_facecolor('#f5f6fa')

        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)

    def update_result_display(self, result):
        """更新当前预测结果的显示（所有类别概率）"""
        print("="*50)
        print("update_result_display 被调用")
        print(f"传入的 result: {result}")

        if not result:
            print("错误: result 为空")
            self.result_label.setText("识别失败：空结果")
            return

        if not result.get('success', False):
            print(f"识别失败，信息: {result.get('message', '未知错误')}")
            self.result_label.setText("识别失败")
            return

        label = result.get('predicted_label', '未知')
        conf = result.get('confidence', 0.0)
        self.result_label.setText(f"{label}\n{conf:.2%}")
        print(f"主标签: {label}, 置信度: {conf:.2%}")

        if conf > 0.8:
            color = "green"
        elif conf > 0.6:
            color = "orange"
        else:
            color = "red"
        self.result_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")

        prob_text = ""
        all_probs = result.get('all_probabilities', {})
        if not all_probs:
            print("警告: all_probabilities 为空")
            prob_text = "无概率数据"
        else:
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            for label, prob in sorted_probs:
                bar_len = int(prob * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                prob_text += f"{label:15} |{bar}| {prob:.2%}\n"

        print("将要设置的概率文本:")
        print(prob_text)

        self.prob_text.setStyleSheet("background-color: white; color: black; font-size: 12px;")
        self.prob_text.setPlainText(prob_text)

        print(f"prob_text 可见性: {self.prob_text.isVisible()}, 大小: {self.prob_text.size()}")

    def start_repeat_from_input(self):
        """从输入框读取参数并开始重复采集，返回是否成功启动"""
        if not self.predictor:
            self.result_label.setText("模型未加载, 无法进行预测")
            self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")
            return False

        try:
            count = int(self.count_edit.text())
            interval_sec = int(self.interval_edit.text())
            if count <= 0 or interval_sec <= 0:
                raise ValueError
        except ValueError:
            self.result_label.setText("请输入有效的正整数")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
            return False

        self.max_collect = count
        self.high_conf_records.clear()
        self.collect_count = 0
        self.timer_collect()  # 立即执行第一次采集
        self.timer.start(interval_sec * 1000)
        return True



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)  # 应用全局样式表
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

