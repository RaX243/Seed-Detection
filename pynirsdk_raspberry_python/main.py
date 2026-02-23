import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLabel, QTextEdit, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette, QColor
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
        self.canvas = None
        

    def initUI(self):

        cerntral = self.centralWidget()
        if cerntral is None:
            cerntral = QWidget()
            self.setCentralWidget(cerntral)
        
        main_layout = QHBoxLayout(cerntral)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        left_panel = QFrame()
        left_panel.setObjectName("card")
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        title = QLabel("控制面板")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(title)

        self.startButton = QPushButton("▶ 开始采集")
        self.startButton.setMinimumHeight(40)
        left_layout.addWidget(self.startButton)
        
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

        right_panel = QFrame()
        right_panel.setObjectName("card")
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)
        
        result_title = QLabel("识别结果")
        result_title.setObjectName("titleLabel")
        right_layout.addWidget(result_title)

        result_card = QFrame()
        result_card.setStyleSheet("background-color: #f0f3f8; border-radius: 5px; padding: 10px;")
        result_card_layout = QVBoxLayout(result_card)
        
        self.result_label = QLabel("等待采集...")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        result_card_layout.addWidget(self.result_label)

        prob_label = QLabel("各类别概率")
        prob_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        right_layout.addWidget(prob_label)
        
        self.prob_text = QTextEdit()
        self.prob_text.setMaximumHeight(300)
        right_layout.addWidget(self.prob_text)
        
        right_layout.addStretch()
        
        right_layout.addWidget(result_card)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(mid_panel, 1)
        main_layout.addWidget(right_panel)

    def load_model(self):
        """加载训练好的模型"""
        try:
            # 找到最新的实验结果文件夹
            results_dir = os.path.join(os.path.dirname(__file__), "data")
            if not os.path.exists(results_dir):
                print("未找到实验结果文件夹")
                return None
            
            exp_folders = [f for f in os.listdir(results_dir) if f.startswith("experiment_")]
            if not exp_folders:
                print("未找到实验文件夹")
                return None
            
            latest_exp = sorted(exp_folders)[-1]   # 获取当前最新的试验数据
            model_path = os.path.join(results_dir, latest_exp, "best_model.pth")  # 模型文件路径
            
            if not os.path.exists(model_path):
                print(f"模型文件不存在: {model_path}")
                return None
            
            # 创建预测器
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
#        data = data.acquire_and_plot_spectrum()  # 这个是数据收集的函数，注释掉之后按照函数的注释自己预设做测试即可，不要删除这个函数
        # 下面的这个数据是预设的，按照函数的注释自己预设做测试即可，在嵌入式设备上测试的时候记得注释掉
        spectrum_data = self.collect_spectrum()

        if spectrum_data['success']:
            print("数据采集成功，波长和强度已更新")
            self.draw_spectrum(spectrum_data['wavelengths'], spectrum_data['intensities'])
            # 开始进行ai预测
            self.recognize_spectrum(spectrum_data)
        else:
            print("数据采集失败:", spectrum_data['message'])
            self.result_label.setText(f"采集失败: {spectrum_data['message']}")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")

    def collect_spectrum(self):
        """采集数据"""
        print("正在采集数据...")
        # ========== 实际采集模式（连接硬件） ==========
        # try:
        #     import data
        #     return data.acquire_and_plot_spectrum()
        # except Exception as e:
        #     print(f"数据采集失败: {e}")
        #     return {
        #         'success': False,
        #         'wavelengths': [],
        #         'intensities': [],
        #         'message': str(e)
        #     }

        # ========== 测试模式：从 Excel 文件随机选取一个样本 ==========
        print("测试模式：从 test/试验数据2.xlsx 随机选取一个样本")
        try:
            # 构建文件路径（相对于当前文件 screen.py 的位置）
            file_path = os.path.join(os.path.dirname(__file__), "test", "试验数据2.xlsx")
            df = pd.read_excel(file_path, header=None)
            
            # 第一行是波长
            wavelengths = df.iloc[0, :-1].tolist()  # 波长列表（float）
            
            # 数据行：第二行到倒数第二列（不包括最后一列标签）
            data_rows = df.iloc[1:, :-1].values.astype(np.float32)
            
            # 随机选择一行
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
            # 降级方案：使用原来的随机生成数据（避免程序崩溃）
            wavelengths = list(range(400, 628))
            intensities = list(np.random.randn(228) * 100 + 500)
            return {
                'success': True,
                'wavelengths': wavelengths,
                'intensities': intensities,
                'message': '文件读取失败，使用随机生成数据'
            }
        

    def recognize_spectrum(self, spectrum_data):
        """
        识别光谱数据并显示结果
        """
        try:
            # 使用模型预测
            result = self.predictor.predict(
                spectrum_data['wavelengths'], 
                spectrum_data['intensities']
            )
            print("预测结果已获取，准备更新显示")  # 新增
            self.update_result_display(result)
            print("update_result_display 调用完成")  # 新增

            
        except Exception as e:
            self.result_label.setText(f"识别失败: {str(e)}")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")


# 这个函数是取消按钮的函数，暂时没什么功能
    def on_change_clicked(self):
        print("change clicked")


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
        print("="*50)
        print("update_result_display 被调用")
        print(f"传入的 result: {result}")
        
        # 检查 result 是否有效
        if not result:
            print("错误: result 为空")
            self.result_label.setText("识别失败：空结果")
            return
        
        if not result.get('success', False):
            print(f"识别失败，信息: {result.get('message', '未知错误')}")
            self.result_label.setText("识别失败")
            return
        
        # 更新主标签
        label = result.get('predicted_label', '未知')
        conf = result.get('confidence', 0.0)
        self.result_label.setText(f"{label}\n{conf:.2%}")
        print(f"主标签: {label}, 置信度: {conf:.2%}")
        
        # 设置置信度颜色
        if conf > 0.8:
            color = "green"
        elif conf > 0.6:
            color = "orange"
        else:
            color = "red"
        self.result_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
        
        # 构建概率文本
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
        
        # 强制设置 prob_text 的样式以确保可见
        self.prob_text.setStyleSheet("background-color: white; color: black; font-size: 12px;")
        self.prob_text.setPlainText(prob_text)
        
        # 额外检查控件是否可见
        print(f"prob_text 可见性: {self.prob_text.isVisible()}, 大小: {self.prob_text.size()}")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)  # 应用全局样式表
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

