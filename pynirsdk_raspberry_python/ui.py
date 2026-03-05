# ui.py (修正版，继承QDialog)
import sys
import os
import json
import time
import datetime
import traceback
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QFormLayout, QLineEdit, QSpinBox,
    QDoubleSpinBox, QPushButton, QTextEdit, QFileDialog,
    QMessageBox, QLabel, QProgressBar, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor

from CNN_Transformer import Config, run_experiment


class TrainingThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config

    def run(self):
        try:
            def log_callback(msg):
                self.log_signal.emit(msg)

            def progress_callback(epoch):
                self.progress_signal.emit(epoch)

            result = run_experiment(
                self.config,
                log_callback=log_callback,
                progress_callback=progress_callback
            )
            self.finished_signal.emit(result)
        except Exception as e:
            self.log_signal.emit(f"错误: {str(e)}\n{traceback.format_exc()}")


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class CNNTransformerGUI(QDialog):
    def __init__(self, parent=None, style_sheet=None):
        super().__init__(parent)
        self.config = Config()
        self.results = None
        self.training_thread = None
        if style_sheet:
            self.setStyleSheet(style_sheet)
        self.setWindowTitle("CNN-Transformer 模型训练")
        self.setMinimumSize(1200, 800)
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        self.btn_load_config = QPushButton("加载配置")
        self.btn_save_config = QPushButton("保存配置")
        self.btn_load_config.clicked.connect(self.load_config)
        self.btn_save_config.clicked.connect(self.save_config)
        toolbar.addWidget(self.btn_load_config)
        toolbar.addWidget(self.btn_save_config)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # 主分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧配置面板
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        tab_widget = QTabWidget()
        config_layout.addWidget(tab_widget)

        # 1. 数据配置选项卡
        data_tab = QWidget()
        tab_widget.addTab(data_tab, "数据")
        data_layout = QFormLayout(data_tab)

        self.edit_file_path = QLineEdit(self.config.file_path)
        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self.browse_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.edit_file_path)
        file_layout.addWidget(self.btn_browse)
        data_layout.addRow("数据文件路径:", file_layout)

        self.edit_sheet_name = QLineEdit(self.config.sheet_name)
        data_layout.addRow("Sheet名称:", self.edit_sheet_name)

        self.spin_test_size = QDoubleSpinBox()
        self.spin_test_size.setRange(0.1, 0.9)
        self.spin_test_size.setSingleStep(0.05)
        self.spin_test_size.setValue(self.config.test_size)
        data_layout.addRow("测试集比例:", self.spin_test_size)

        self.spin_random_seed = QSpinBox()
        self.spin_random_seed.setRange(1, 10000)
        self.spin_random_seed.setValue(self.config.random_seed)
        data_layout.addRow("随机种子:", self.spin_random_seed)

        # 2. 训练参数选项卡
        train_tab = QWidget()
        tab_widget.addTab(train_tab, "训练")
        train_layout = QFormLayout(train_tab)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(self.config.epochs)
        train_layout.addRow("训练轮数:", self.spin_epochs)

        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 256)
        self.spin_batch_size.setValue(self.config.batch_size)
        train_layout.addRow("批次大小:", self.spin_batch_size)

        self.spin_learning_rate = QDoubleSpinBox()
        self.spin_learning_rate.setRange(1e-6, 1e-1)
        self.spin_learning_rate.setDecimals(6)
        self.spin_learning_rate.setSingleStep(1e-5)
        self.spin_learning_rate.setValue(self.config.learning_rate)
        train_layout.addRow("学习率:", self.spin_learning_rate)

        self.spin_weight_decay = QDoubleSpinBox()
        self.spin_weight_decay.setRange(0, 1e-2)
        self.spin_weight_decay.setDecimals(6)
        self.spin_weight_decay.setSingleStep(1e-6)
        self.spin_weight_decay.setValue(self.config.weight_decay)
        train_layout.addRow("L2正则化:", self.spin_weight_decay)

        self.spin_l1_lambda = QDoubleSpinBox()
        self.spin_l1_lambda.setRange(0, 1e-2)
        self.spin_l1_lambda.setDecimals(6)
        self.spin_l1_lambda.setSingleStep(1e-6)
        self.spin_l1_lambda.setValue(self.config.l1_lambda)
        train_layout.addRow("L1正则化:", self.spin_l1_lambda)

        self.spin_leaky_relu = QDoubleSpinBox()
        self.spin_leaky_relu.setRange(0.001, 0.1)
        self.spin_leaky_relu.setSingleStep(0.001)
        self.spin_leaky_relu.setValue(self.config.leaky_relu_slope)
        train_layout.addRow("LeakyReLU斜率:", self.spin_leaky_relu)

        # 3. 模型参数选项卡
        model_tab = QWidget()
        tab_widget.addTab(model_tab, "模型")
        model_layout = QFormLayout(model_tab)

        self.spin_embed_dim = QSpinBox()
        self.spin_embed_dim.setRange(32, 512)
        self.spin_embed_dim.setSingleStep(16)
        self.spin_embed_dim.setValue(self.config.embed_dim)
        model_layout.addRow("嵌入维度:", self.spin_embed_dim)

        self.spin_num_heads = QSpinBox()
        self.spin_num_heads.setRange(1, 16)
        self.spin_num_heads.setValue(self.config.num_heads)
        model_layout.addRow("注意力头数:", self.spin_num_heads)

        self.spin_ff_dim = QSpinBox()
        self.spin_ff_dim.setRange(128, 2048)
        self.spin_ff_dim.setSingleStep(64)
        self.spin_ff_dim.setValue(self.config.ff_dim)
        model_layout.addRow("前馈网络维度:", self.spin_ff_dim)

        self.spin_num_layers = QSpinBox()
        self.spin_num_layers.setRange(1, 6)
        self.spin_num_layers.setValue(self.config.num_layers)
        model_layout.addRow("Transformer层数:", self.spin_num_layers)

        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 0.5)
        self.spin_dropout.setSingleStep(0.05)
        self.spin_dropout.setValue(self.config.dropout_rate)
        model_layout.addRow("Dropout率:", self.spin_dropout)

        self.spin_fusion_dropout = QDoubleSpinBox()
        self.spin_fusion_dropout.setRange(0.0, 0.5)
        self.spin_fusion_dropout.setSingleStep(0.05)
        self.spin_fusion_dropout.setValue(self.config.fusion_dropout)
        model_layout.addRow("融合层Dropout:", self.spin_fusion_dropout)

        self.spin_local_window = QSpinBox()
        self.spin_local_window.setRange(1, 50)
        self.spin_local_window.setValue(self.config.local_window_size)
        model_layout.addRow("局部窗口大小:", self.spin_local_window)

        self.spin_global_window = QSpinBox()
        self.spin_global_window.setRange(1, 100)
        self.spin_global_window.setValue(self.config.global_window_size)
        model_layout.addRow("全局窗口大小:", self.spin_global_window)

        self.spin_k_bands = QSpinBox()
        self.spin_k_bands.setRange(1, 30)
        self.spin_k_bands.setValue(self.config.k_bands)
        model_layout.addRow("关键波段数量:", self.spin_k_bands)

        self.spin_cnn_filters = QSpinBox()
        self.spin_cnn_filters.setRange(16, 256)
        self.spin_cnn_filters.setSingleStep(16)
        self.spin_cnn_filters.setValue(self.config.cnn_filters)
        model_layout.addRow("CNN滤波器数:", self.spin_cnn_filters)

        self.spin_cnn_kernel = QSpinBox()
        self.spin_cnn_kernel.setRange(1, 7)
        self.spin_cnn_kernel.setValue(self.config.cnn_kernel_size)
        model_layout.addRow("CNN核大小:", self.spin_cnn_kernel)

        # 4. 结果保存选项卡
        save_tab = QWidget()
        tab_widget.addTab(save_tab, "保存")
        save_layout = QFormLayout(save_tab)

        self.edit_results_dir = QLineEdit(self.config.results_dir)
        self.btn_browse_dir = QPushButton("浏览...")
        self.btn_browse_dir.clicked.connect(self.browse_dir)
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.edit_results_dir)
        dir_layout.addWidget(self.btn_browse_dir)
        save_layout.addRow("结果保存目录:", dir_layout)

        # 启动按钮
        self.btn_start = QPushButton("开始训练")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_training)
        config_layout.addWidget(self.btn_start)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        config_layout.addWidget(self.progress_bar)

        splitter.addWidget(config_widget)

        # 右侧：日志和图表
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        right_layout.addWidget(QLabel("训练日志:"))
        right_layout.addWidget(self.log_text)

        chart_tabs = QTabWidget()
        self.canvas_loss = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas_acc = MplCanvas(self, width=5, height=4, dpi=100)
        chart_tabs.addTab(self.canvas_loss, "损失曲线")
        chart_tabs.addTab(self.canvas_acc, "准确率曲线")
        right_layout.addWidget(chart_tabs)

        splitter.addWidget(right_widget)
        splitter.setSizes([500, 700])
        main_layout.addWidget(splitter)

        # 状态标签（替代状态栏）
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 5, 5, 5)
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: gray;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        main_layout.addWidget(status_frame)

    # 以下方法保持不变
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "Excel files (*.xlsx *.xls)")
        if file_path:
            self.edit_file_path.setText(file_path)

    def browse_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择结果保存目录")
        if dir_path:
            self.edit_results_dir.setText(dir_path)

    def load_config(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "加载配置文件", "", "JSON files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config_dict = json.load(f)
                self.edit_file_path.setText(config_dict.get('file_path', self.config.file_path))
                self.edit_sheet_name.setText(config_dict.get('sheet_name', self.config.sheet_name))
                self.spin_test_size.setValue(config_dict.get('test_size', self.config.test_size))
                self.spin_random_seed.setValue(config_dict.get('random_seed', self.config.random_seed))
                self.spin_epochs.setValue(config_dict.get('epochs', self.config.epochs))
                self.spin_batch_size.setValue(config_dict.get('batch_size', self.config.batch_size))
                self.spin_learning_rate.setValue(config_dict.get('learning_rate', self.config.learning_rate))
                self.spin_weight_decay.setValue(config_dict.get('weight_decay', self.config.weight_decay))
                self.spin_l1_lambda.setValue(config_dict.get('l1_lambda', self.config.l1_lambda))
                self.spin_leaky_relu.setValue(config_dict.get('leaky_relu_slope', self.config.leaky_relu_slope))
                self.spin_embed_dim.setValue(config_dict.get('embed_dim', self.config.embed_dim))
                self.spin_num_heads.setValue(config_dict.get('num_heads', self.config.num_heads))
                self.spin_ff_dim.setValue(config_dict.get('ff_dim', self.config.ff_dim))
                self.spin_num_layers.setValue(config_dict.get('num_layers', self.config.num_layers))
                self.spin_dropout.setValue(config_dict.get('dropout_rate', self.config.dropout_rate))
                self.spin_fusion_dropout.setValue(config_dict.get('fusion_dropout', self.config.fusion_dropout))
                self.spin_local_window.setValue(config_dict.get('local_window_size', self.config.local_window_size))
                self.spin_global_window.setValue(config_dict.get('global_window_size', self.config.global_window_size))
                self.spin_k_bands.setValue(config_dict.get('k_bands', self.config.k_bands))
                self.spin_cnn_filters.setValue(config_dict.get('cnn_filters', self.config.cnn_filters))
                self.spin_cnn_kernel.setValue(config_dict.get('cnn_kernel_size', self.config.cnn_kernel_size))
                self.edit_results_dir.setText(config_dict.get('results_dir', self.config.results_dir))
                self.log("配置加载成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载配置失败: {str(e)}")

    def save_config(self):
        self.update_config_from_ui()
        config_dict = {k: v for k, v in vars(self.config).items() if not k.startswith('__') and not callable(v)}
        file_path, _ = QFileDialog.getSaveFileName(self, "保存配置文件", "", "JSON files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config_dict, f, indent=4)
                self.log("配置保存成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存配置失败: {str(e)}")

    def update_config_from_ui(self):
        self.config.file_path = self.edit_file_path.text()
        self.config.sheet_name = self.edit_sheet_name.text()
        self.config.test_size = self.spin_test_size.value()
        self.config.random_seed = self.spin_random_seed.value()
        self.config.epochs = self.spin_epochs.value()
        self.config.batch_size = self.spin_batch_size.value()
        self.config.learning_rate = self.spin_learning_rate.value()
        self.config.weight_decay = self.spin_weight_decay.value()
        self.config.l1_lambda = self.spin_l1_lambda.value()
        self.config.leaky_relu_slope = self.spin_leaky_relu.value()
        self.config.embed_dim = self.spin_embed_dim.value()
        self.config.num_heads = self.spin_num_heads.value()
        self.config.ff_dim = self.spin_ff_dim.value()
        self.config.num_layers = self.spin_num_layers.value()
        self.config.dropout_rate = self.spin_dropout.value()
        self.config.fusion_dropout = self.spin_fusion_dropout.value()
        self.config.local_window_size = self.spin_local_window.value()
        self.config.global_window_size = self.spin_global_window.value()
        self.config.k_bands = self.spin_k_bands.value()
        self.config.cnn_filters = self.spin_cnn_filters.value()
        self.config.cnn_kernel_size = self.spin_cnn_kernel.value()
        self.config.results_dir = self.edit_results_dir.text()

    def log(self, message):
        self.log_text.append(message)
        self.log_text.moveCursor(QTextCursor.End)

    def start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "警告", "训练正在进行中，请等待完成")
            return

        self.update_config_from_ui()

        if not os.path.exists(self.config.file_path):
            QMessageBox.critical(self, "错误", "数据文件不存在")
            return

        self.log_text.clear()
        self.log("开始训练...")
        self.log(f"配置: {vars(self.config)}")

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self.config.epochs)
        self.progress_bar.setValue(0)

        self.training_thread = TrainingThread(self.config)
        self.training_thread.log_signal.connect(self.log)
        self.training_thread.progress_signal.connect(self.progress_bar.setValue)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()

        self.btn_start.setEnabled(False)
        self.status_label.setText("训练中...")
        self.status_label.setStyleSheet("color: orange;")

    def training_finished(self, results):
        self.results = results
        self.progress_bar.setVisible(False)
        self.btn_start.setEnabled(True)
        self.status_label.setText("训练完成")
        self.status_label.setStyleSheet("color: green;")

        self.log("\n=== 训练完成 ===")
        self.log(f"测试集准确率: {results.get('test_accuracy', 'N/A')}")
        self.log(f"测试集Kappa: {results.get('test_kappa', 'N/A')}")

        if 'train_losses' in results and 'test_losses' in results:
            self.plot_curves(results)

        QMessageBox.information(self, "完成", "训练和评估完成！结果已保存。")

    def plot_curves(self, results):
        self.canvas_loss.axes.clear()
        epochs = range(1, len(results['train_losses']) + 1)
        self.canvas_loss.axes.plot(epochs, results['train_losses'], 'b-', label='训练损失')
        self.canvas_loss.axes.plot(epochs, results['test_losses'], 'r-', label='测试损失')
        self.canvas_loss.axes.set_xlabel('Epoch')
        self.canvas_loss.axes.set_ylabel('Loss')
        self.canvas_loss.axes.legend()
        self.canvas_loss.axes.grid(True)
        self.canvas_loss.draw()

        self.canvas_acc.axes.clear()
        self.canvas_acc.axes.plot(epochs, results['train_accuracies'], 'b-', label='训练准确率')
        self.canvas_acc.axes.plot(epochs, results['test_accuracies'], 'r-', label='测试准确率')
        self.canvas_acc.axes.set_xlabel('Epoch')
        self.canvas_acc.axes.set_ylabel('Accuracy (%)')
        self.canvas_acc.axes.legend()
        self.canvas_acc.axes.grid(True)
        self.canvas_acc.draw()


def run_gui():
    app = QApplication(sys.argv)
    window = CNNTransformerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    run_gui()