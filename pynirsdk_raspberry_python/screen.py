import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt6 import uic
Form, Window = uic.loadUiType("mainwindow.ui")
# import data   # 导入数据采集模块， 本机测试的时候记得注释掉， 如果在嵌入式设备上测试就取消注释
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import font_manager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os


# 设置中文字体
def setup_chinese_font():
    preferred = ['noto', 'wqy', 'simhei', 'msyh', 'msjh', 'arphic', 'ukai']
    found_font = None
    font_paths = (font_manager.findSystemFonts(fontpaths=None, fontext='ttf') + 
                  font_manager.findSystemFonts(fontpaths=None, fontext='otf'))
    for fpath in font_paths:
        name = os.path.basename(fpath).lower()
        if any(p in name for p in preferred):
            try:
                prop = font_manager.FontProperties(fname=fpath)
                found_font = prop.get_name()
                break
            except Exception:
                continue
    
    if found_font:
        matplotlib.rcParams['font.sans-serif'] = [found_font]
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

# 预设，更改代码时候记得不要删除这个函数，保持中文显示正常
setup_chinese_font()

class MainWindow(Window, Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.dlp = None

        self.setWindowTitle("光谱仪控制界面")

        self.startButton.clicked.connect(self.on_start_clicked)
        self.cancalButton.clicked.connect(self.on_change_clicked)

        if hasattr(self, 'plot_widget'):
            self.plot_widget = self.plot_widget
        else:
            self.plot_widget = QWidget(self)
            self.plot_widget.setObjectName("plot_widget")
            self.plot_widget.setGeometry(200, 80, 600, 400) # 自己调整位置和大小
            self.layout = QVBoxLayout()
            self.plot_widget.setLayout(self.layout)
            self.centralWidget().layout().addWidget(self.plot_widget)

        self.plot_layout = QVBoxLayout()
        self.plot_widget.setLayout(self.plot_layout)
        
        # 保存canvas引用，避免重复创建
        self.canvas = None

# 这个函数是开始按钮的函数，加载数据显示用的，更改代码的时候不要更改函数名
    def on_start_clicked(self):
        print("start clicked")
#        data = data.acquire_and_plot_spectrum()  # 这个是数据收集的函数，注释掉之后按照函数的注释自己预设做测试即可，不要删除这个函数
        # 下面的这个数据是预设的，按照函数的注释自己预设做测试即可，在嵌入式设备上测试的时候记得注释掉
        data = {
            'success': True,
            'wavelengths': [400, 500, 600, 700, 800],
            'intensities': [10, 25, 35, 45, 55],
            'message': ''
        }

        if data['success']:
            print("数据采集成功，波长和强度已更新")
            self.draw_spectrum(data['wavelengths'], data['intensities'])
        else:
            print("数据采集失败:", data['message'])
    
# 这个函数是取消按钮的函数，暂时没什么功能
    def on_change_clicked(self):
        print("change clicked")
        
    def draw_spectrum(self, wavelengths, intensities):
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(wavelengths, intensities, marker='o', linestyle='-', color='blue')
        ax.set_title("光谱图")
        ax.set_xlabel("波长 (nm)")
        ax.set_ylabel("强度")
        ax.grid()

        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)

if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建应用程序
    window = MainWindow()          # 创建主窗口
    window.show()                  # 显示窗口
    sys.exit(app.exec())  