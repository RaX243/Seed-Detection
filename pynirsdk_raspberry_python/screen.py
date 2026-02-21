import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QTextEdit
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
from predictor import SpectrumPredictor



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


        self.predictor = self.load_model()
        self.create_results_display()


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

        if self.predictor:
            self.labels.setText("✓ 模型已加载")
        else:
            self.labels.setText("✗ 模型加载失败")




# 这个函数是开始按钮的函数，加载数据显示用的，更改代码的时候不要更改函数名
    def on_start_clicked(self):
        print("start clicked")
#        data = data.acquire_and_plot_spectrum()  # 这个是数据收集的函数，注释掉之后按照函数的注释自己预设做测试即可，不要删除这个函数
        # 下面的这个数据是预设的，按照函数的注释自己预设做测试即可，在嵌入式设备上测试的时候记得注释掉
        spectrum_data = self.collect_spectrum()

        if not self.predictor:
            print("无法进行预测，因为模型未加载")
            return
        

        if spectrum_data['success']:
            print("数据采集成功，波长和强度已更新")
            self.draw_spectrum(spectrum_data['wavelengths'], spectrum_data['intensities'])

            # 开始进行ai预测
            self.recognize_spectrum(spectrum_data)



        else:
            print("数据采集失败:", spectrum_data['message'])
            self.result_label.setText(f"采集失败: {spectrum_data['message']}")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
    
# 这个函数是取消按钮的函数，暂时没什么功能
    def on_change_clicked(self):
        print("change clicked")



    def create_results_display(self):
        # 预测结果标签
        self.result_label = QLabel("等待采集数据...", self)
        self.result_label.setGeometry(820, 80, 300, 50)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        
        # 概率显示区域
        self.prob_text = QTextEdit(self)
        self.prob_text.setGeometry(820, 140, 300, 250)
        self.prob_text.setReadOnly(True)
        self.prob_text.setStyleSheet("font-size: 12px;")
        
        # 模型状态标签
        self.labels = QLabel("正在查找模型...", self)
        self.labels.setGeometry(820, 400, 300, 30)
        self.labels.setStyleSheet("font-size: 12px; color: gray;")



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


    def draw_spectrum(self, wavelengths, intensities):
        """绘制光谱图"""
        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(wavelengths, intensities, 'b-', linewidth=1.5)
        ax.set_title("实时光谱图")
        ax.set_xlabel("波长 (nm)")
        ax.set_ylabel("强度")
        ax.grid(True, alpha=0.3)
        
        # 如果有预测结果，在图上标注
        if hasattr(self, 'result_label') and self.result_label.text() != "等待采集数据...":
            ax.text(0.02, 0.98, self.result_label.text().split('\n')[0], 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        self.canvas = FigureCanvas(fig)
        self.plot_layout.addWidget(self.canvas)



    def collect_spectrum(self):
        """采集数据"""
        print("正在采集数据...")
        # 下面的是实际测量的函数 正常使用的时候取消注释，把我们随机生成的注释
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

        # 下面的代码是测试用的，是预设的，如果使用实际测试请把下面的代码注释
        print("使用随机假数据")
        wavelengths = list(range(400, 628))  # 228个波长点
        intensities = list(np.random.randn(228) * 100 + 500)  # 228个强度值
        
        return {
            'success': True,
            'wavelengths': wavelengths,
            'intensities': intensities,
            'message': '测试数据采集成功'
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
            
            # 更新结果显示
            self.update_result_display(result)
            
        except Exception as e:
            print(f"识别失败: {e}")
            self.result_label.setText(f"识别失败: {str(e)}")
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")



    def update_result_display(self, result):
        """
        更新结果显示
        """
        if not result['success']:
            self.result_label.setText(f"识别失败: {result.get('error', '未知错误')}")
            return
        
        # 显示主要结果
        label_text = f"预测结果: {result['predicted_label']}\n置信度: {result['confidence']:.2%}"
        self.result_label.setText(label_text)
        
        # 根据置信度设置颜色
        if result['confidence'] > 0.8:
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        elif result['confidence'] > 0.6:
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: orange;")
        else:
            self.result_label.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
        
        # 显示详细概率
        prob_text = "各类别概率:\n" + "="*25 + "\n"
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        
        for label, prob in sorted_probs:
            # 创建进度条效果
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            prob_text += f"{label:15} |{bar}| {prob:.2%}\n"
        
        self.prob_text.setText(prob_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)  # 创建应用程序
    window = MainWindow()          # 创建主窗口
    window.show()                  # 显示窗口
    sys.exit(app.exec())  