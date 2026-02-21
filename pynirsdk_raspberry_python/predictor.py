import torch
import torch.nn as nn
import numpy as np
import os
import json
from CNN_Transformer import FeatureConcatenationFusionWithDual, Config

class SpectrumPredictor:
    """光谱预测器类"""
    
    def __init__(self, model_path, config=None):
        """
        初始化预测器
        
        参数:
            model_path: 模型文件路径（如 '实验结果/experiment_xxx/best_model.pth'）
            config: 配置参数（可选）
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型和配置
        self.model, self.label_mapping, self.idx_to_label = self.load_model(model_path, config)
        self.model.eval()
        
        print(f"模型加载成功！")
        print(f"类别数量: {len(self.label_mapping)}")
        print(f"类别映射: {self.label_mapping}")
    
    def load_model(self, model_path, config=None):
        """加载训练好的模型"""
        # 获取模型所在目录
        model_dir = os.path.dirname(model_path)
        
        # 加载标签映射（如果存在）
        label_mapping_file = os.path.join(model_dir, 'label_mapping.json')
        if os.path.exists(label_mapping_file):
            with open(label_mapping_file, 'r') as f:
                label_mapping = json.load(f)
        else:
            # 如果没有标签映射文件，使用默认映射
            label_mapping = {'Unknown': 0}
        
        # 创建反向映射
        idx_to_label = {v: k for k, v in label_mapping.items()}
        
        # 确定输入维度（从模型文件推断）
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取输入大小（这里需要根据你的数据调整）
        # 如果checkpoint中包含输入大小信息
        if 'input_size' in checkpoint:
            input_size = checkpoint['input_size']
        else:
            # 默认使用228（根据你的数据）
            input_size = 228
        
        # 创建模型实例
        num_classes = len(label_mapping)
        model = FeatureConcatenationFusionWithDual(
            input_size=input_size, 
            num_classes=num_classes
        )
        
        # 加载模型权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        
        return model, label_mapping, idx_to_label
    
    def preprocess_spectrum(self, wavelengths, intensities):
        """
        预处理光谱数据，使其符合模型输入要求
        
        参数:
            wavelengths: 波长列表
            intensities: 强度列表
        
        返回:
            torch.Tensor: 预处理后的数据
        """
        # 转换为numpy数组
        intensities = np.array(intensities, dtype=np.float32)
        
        # 检查是否需要插值（如果采集的数据点数和模型训练时不一致）
        if len(intensities) != 228:  # 假设模型训练时用了228个点
            # 这里需要根据你的实际情况进行插值或截断
            # 简单示例：如果点数多就截取，少就填充
            if len(intensities) > 228:
                intensities = intensities[:228]
            else:
                # 填充0（或者可以用插值）
                padded = np.zeros(228)
                padded[:len(intensities)] = intensities
                intensities = padded
        
        # 归一化（根据训练时的预处理方式）
        # 这里需要根据你的训练数据预处理方式调整
        intensities = (intensities - np.mean(intensities)) / (np.std(intensities) + 1e-8)
        
        # 转换为torch张量，添加batch和channel维度
        tensor = torch.from_numpy(intensities).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 228]
        tensor = tensor.to(self.device)
        
        return tensor
    
    def predict(self, wavelengths, intensities):
        """
        预测光谱类别
        
        参数:
            wavelengths: 波长列表
            intensities: 强度列表
        
        返回:
            dict: 预测结果
        """
        # 预处理
        input_tensor = self.preprocess_spectrum(wavelengths, intensities)
        
        # 推理
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(outputs, dim=1).item()
        
        # 获取预测结果
        predicted_label = self.idx_to_label.get(predicted_idx, f"未知类别_{predicted_idx}")
        confidence = probabilities[0][predicted_idx].item()
        
        # 获取所有类别的概率
        all_probs = {}
        for idx, prob in enumerate(probabilities[0]):
            label = self.idx_to_label.get(idx, f"未知类别_{idx}")
            all_probs[label] = prob.item()
        
        return {
            'success': True,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    
    def predict_batch(self, spectra_list):
        """
        批量预测
        
        参数:
            spectra_list: 光谱数据列表，每个元素是 (wavelengths, intensities)
        
        返回:
            list: 预测结果列表
        """
        results = []
        for wavelengths, intensities in spectra_list:
            result = self.predict(wavelengths, intensities)
            results.append(result)
        return results


# 简单的测试函数
def test_predictor():
    """测试预测器"""
    # 找到最新的实验结果文件夹
    results_dir = os.path.join(os.path.dirname(__file__), "实验结果")
    if not os.path.exists(results_dir):
        print("未找到实验结果文件夹")
        return
    
    # 获取最新的实验文件夹
    exp_folders = [f for f in os.listdir(results_dir) if f.startswith("experiment_")]
    if not exp_folders:
        print("未找到实验文件夹")
        return
    
    latest_exp = sorted(exp_folders)[-1]
    model_path = os.path.join(results_dir, latest_exp, "best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 创建预测器
    predictor = SpectrumPredictor(model_path)
    
    # 测试数据（用一些随机数据）
    test_wavelengths = list(range(400, 628, 1))  # 228个点
    test_intensities = np.random.randn(228) * 100 + 500
    
    # 预测
    result = predictor.predict(test_wavelengths, test_intensities)
    
    print("\n测试预测结果:")
    print(f"预测类别: {result['predicted_label']}")
    print(f"置信度: {result['confidence']:.4f}")
    print("所有类别概率:")
    for label, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    test_predictor()