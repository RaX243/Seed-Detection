from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import ctypes
import wrapper
import os
import sys
import time
import json
import numpy as np
import glob  # 新增
from predictor import SpectrumPredictor

app = Flask(__name__, static_folder='static')
CORS(app)

# 全局变量
active_device = -1
PIXEL_NUM = 228
WLS_NUM = PIXEL_NUM

# 预测器全局变量
predictor = None

def find_latest_model():
    """
    按照 main.py 的方式查找最新的模型：
    在 data 文件夹下查找 experiment_ 开头的文件夹
    """
    # 获取 data 文件夹路径
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    if not os.path.exists(data_dir):
        print(f"data 文件夹不存在: {data_dir}")
        return None
    
    # 查找 experiment_ 开头的文件夹
    exp_folders = [f for f in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, f)) and f.startswith("experiment_")]
    
    if not exp_folders:
        print("未找到 experiment_ 开头的文件夹")
        return None
    
    # 按名称排序，取最新的
    latest_exp = sorted(exp_folders)[-1]
    model_path = os.path.join(data_dir, latest_exp, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return None
    
    print(f"找到最新模型: {model_path}")
    return model_path

def load_prediction_model():
    """加载预测模型"""
    global predictor
    try:
        model_path = find_latest_model()
        if model_path:
            print(f"加载预测模型: {model_path}")
            predictor = SpectrumPredictor(model_path)
            print("预测模型加载成功！")
            if predictor:
                print(f"类别数量: {len(predictor.label_mapping)}")
                print(f"类别映射: {predictor.label_mapping}")
        else:
            print("未找到合适的模型文件")
    except Exception as e:
        print(f"加载预测模型失败: {e}")
        import traceback
        traceback.print_exc()

# 启动时加载模型
load_prediction_model()

# ... 其他路由保持不变（index, connect, get_wavelengths, get_intensities, get_info）...

# 提供静态页面
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# API: 连接设备
@app.route('/api/connect', methods=['POST'])
def connect():
    global active_device
    try:
        print("尝试连接设备...")
        count = wrapper.dlpConnect()
        print(f"找到 {count} 个设备")
        
        if count > 0:
            ret = wrapper.dlpOpenByUsb(0)
            if ret >= 0:
                active_device = 0
                # 延时一下让设备稳定
                time.sleep(0.5)
                return jsonify({'success': True, 'message': f'已连接 {count} 个设备，打开设备0'})
            else:
                return jsonify({'success': False, 'message': f'打开设备失败，错误码：{ret}'})
        else:
            return jsonify({'success': False, 'message': '未找到设备'})
    except Exception as e:
        print(f"连接设备异常：{str(e)}")
        return jsonify({'success': False, 'message': f'异常：{str(e)}'})

# API: 获取波长
@app.route('/api/wavelengths', methods=['GET'])
def get_wavelengths():
    global active_device
    if active_device < 0:
        return jsonify({'success': False, 'message': '请先连接设备'})
    
    try:
        # 使用正确的像素数 228
        wls = (ctypes.c_double * PIXEL_NUM)()
        ret = wrapper.dlpGetWavelengths(wls, PIXEL_NUM)
        
        if ret >= 0:
            wavelengths = [wls[i] for i in range(PIXEL_NUM)]
            # 打印前几个波长用于调试
            print(f"波长前5个: {[f'{w:.2f}' for w in wavelengths[:5]]}")
            return jsonify({'success': True, 'wavelengths': wavelengths})
        else:
            return jsonify({'success': False, 'message': f'获取波长失败，错误码：{ret}'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'异常：{str(e)}'})

# API: 获取强度
@app.route('/api/intensities', methods=['GET'])
def get_intensities():
    global active_device
    if active_device < 0:
        return jsonify({'success': False, 'message': '请先连接设备'})
    
    try:
        intensities = (ctypes.c_int * PIXEL_NUM)()
        ret = wrapper.dlpGetIntensities(active_device, intensities, PIXEL_NUM)
        
        if ret >= 0:
            intensity_list = [intensities[i] for i in range(PIXEL_NUM)]
            # 打印前几个强度用于调试
            print(f"强度前5个: {intensity_list[:5]}")
            return jsonify({'success': True, 'intensities': intensity_list})
        else:
            return jsonify({'success': False, 'message': f'获取强度失败，错误码：{ret}'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'异常：{str(e)}'})

# API: 获取设备信息
@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify({
        'success': True,
        'device_active': active_device >= 0,
        'pixel_num': PIXEL_NUM,
        'message': f'设备{"已连接" if active_device >= 0 else "未连接"}'
    })


# 修改预测端点，添加更多调试信息
@app.route('/api/predict', methods=['POST'])
def predict():
    """接收光谱数据并返回预测结果"""
    global predictor, active_device
    
    # 检查模型是否加载
    if predictor is None:
        return jsonify({
            'success': False,
            'message': '预测模型未加载，请检查 data 文件夹下是否有 experiment_ 开头的文件夹'
        })
    
    # 检查设备是否连接
    if active_device < 0:
        return jsonify({
            'success': False, 
            'message': '请先连接光谱仪设备'
        })
    
    try:
        # 获取最新的强度数据
        intensities = (ctypes.c_int * PIXEL_NUM)()
        ret = wrapper.dlpGetIntensities(active_device, intensities, PIXEL_NUM)
        
        if ret < 0:
            return jsonify({
                'success': False,
                'message': f'获取强度数据失败，错误码：{ret}'
            })
        
        # 转换为列表
        intensity_list = [intensities[i] for i in range(PIXEL_NUM)]
        
        # 获取波长数据
        wls = (ctypes.c_double * PIXEL_NUM)()
        wls_ret = wrapper.dlpGetWavelengths(wls, PIXEL_NUM)
        
        if wls_ret >= 0:
            wavelength_list = [wls[i] for i in range(PIXEL_NUM)]
        else:
            # 如果获取波长失败，生成默认波长
            wavelength_list = list(range(400, 400 + PIXEL_NUM))
            print(f"使用默认波长: {wavelength_list[:5]}...")
        
        # 进行预测
        result = predictor.predict(wavelength_list, intensity_list)
        
        # 添加调试信息
        print(f"预测结果: {result['predicted_label']} (置信度: {result['confidence']:.2%})")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"预测出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'预测过程出错：{str(e)}'
        })

# 新增：获取模型信息
@app.route('/api/model/info', methods=['GET'])
def model_info():
    """返回模型的基本信息（与 main.py 保持一致）"""
    global predictor
    
    if predictor is None:
        return jsonify({
            'success': False,
            'message': '模型未加载',
            'model_path': None,
            'num_classes': 0,
            'classes': []
        })
    
    return jsonify({
        'success': True,
        'model_path': find_latest_model(),
        'num_classes': len(predictor.label_mapping),
        'classes': list(predictor.label_mapping.keys()),
        'device': str(predictor.device)
    })

# 新增：列出所有可用的实验文件夹（用于调试）
@app.route('/api/model/list', methods=['GET'])
def list_models():
    """列出 data 文件夹下所有的实验文件夹"""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    if not os.path.exists(data_dir):
        return jsonify({
            'success': False,
            'message': f'data 文件夹不存在: {data_dir}'
        })
    
    exp_folders = [f for f in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, f)) and f.startswith("experiment_")]
    
    models = []
    for folder in sorted(exp_folders, reverse=True):
        model_path = os.path.join(data_dir, folder, "best_model.pth")
        if os.path.exists(model_path):
            models.append({
                'folder': folder,
                'path': model_path,
                'exists': True
            })
        else:
            models.append({
                'folder': folder,
                'path': model_path,
                'exists': False
            })
    
    return jsonify({
        'success': True,
        'data_dir': data_dir,
        'models': models
    })

if __name__ == '__main__':
    print(f"启动服务器...")
    print(f"像素点数: {PIXEL_NUM}")
    print(f"data目录: {os.path.join(os.path.dirname(__file__), 'data')}")
    print(f"模型已加载: {predictor is not None}")
    if predictor:
        print(f"类别数量: {len(predictor.label_mapping)}")
    print(f"静态文件目录: {os.path.abspath('static')}")
    print(f"访问地址: http://10.66.240.222:5000")
    app.run(host='10.66.240.222', port=5000, debug=True, threaded=True)
