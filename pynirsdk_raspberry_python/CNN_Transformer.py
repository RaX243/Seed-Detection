import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib
import os
import datetime
import builtins
import json
import time
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

class Config:
    """模型配置参数"""
    # 路径重写
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 获取当前文件所在目录
    DATA_DIR = os.path.join(BASE_DIR, '试验数据2预处理') # 数据目录,在引号之间更改
    file_path = os.path.join(DATA_DIR, '标准正态变量交换snvdata_snv.xlsx') # 数据文件路径,在引号之间更改
    sheet_name = 'Sheet3'  # Excel表格的Sheet名称
    # 训练参数
    epochs = 200
    batch_size = 12
    learning_rate = 0.00007
    weight_decay = 1e-5
    l1_lambda = 1e-6
    leaky_relu_slope = 0.01  # LeakyReLU的斜率

    # Transformer参数
    embed_dim = 128  # 嵌入维度
    num_heads = 8  # 注意力头数
    ff_dim = 512  # 前馈网络维度
    num_layers = 3  # Transformer层数
    dropout_rate = 0.3  # Dropout比例
    fusion_dropout = 0.5  # 融合层Dropout

    # 注意力参数
    local_window_size = 10  # 局部注意力窗口大小
    global_window_size = 20  # 全局注意力窗口大小
    k_bands = 10  # 自适应物理注意力关注的关键波段数量

    # 1D-CNN参数
    cnn_filters = 64  # 第一层CNN的滤波器数量
    cnn_kernel_size = 3  # CNN核大小

    # 数据集参数
    test_size = 0.3
    random_seed = 42

    # 结果保存路径
    results_dir = r"C:\Users\Lenovo\Desktop\实验数据：数据整理\YM模型数据保存\实验数据YM 1000\AAA\CNN_Transformer"  # 替换为您的保存路径


# 读取 Excel 数据
def load_data(file_path, sheet_name):
    """加载并预处理数据"""
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    wavelengths = df.iloc[0, :-1].values  # 波长点
    data = df.iloc[1:, :-1].values.astype(np.float32)  # 光谱数据
    labels = df.iloc[1:, -1].values  # 标签，保持原始格式

    print(f"\n数据加载完成:")
    print(f"波长点数: {len(wavelengths)}")
    print(f"样本数量: {len(data)}")
    print(f"类别数量: {len(np.unique(labels))}")
    print(f"波长范围: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f}")
    print(f"数据范围: {np.min(data):.4f} - {np.max(data):.4f}")

    return data, labels, wavelengths

# ======================================
class SpectralDataset(Dataset):
    """光谱数据集"""
    def __init__(self, data, labels, label_to_idx):
        self.data = data.astype(np.float32)
        # 转换为整数索引
        self.labels = [label_to_idx[l] for l in labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return torch.tensor(x).unsqueeze(0), torch.tensor(y)

class PositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, max_len=2000):
        super().__init__()
        self.position_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.position_emb, std=0.02)

    def forward(self, x):
        return x + self.position_emb[:, :x.size(1), :]


class StandardTransformerLayer(nn.Module):
    """标准Transformer层（多头注意力+前馈网络）"""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 多头注意力
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # 前馈网络
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class LocalSparseAttention(nn.Module):
    """局部稀疏注意力（可配置窗口大小）"""

    def __init__(self, embed_dim, num_heads, window_size=10):
        super().__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, L, D = x.shape
        # 创建局部注意力掩码
        mask = torch.ones(L, L, dtype=torch.bool, device=x.device)
        for i in range(L):
            start = max(0, i - self.window_size // 2)
            end = min(L, i + self.window_size // 2)
            mask[i, :start] = False
            mask[i, end:] = False

        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)


class AdaptivePhysicallyConstrainedAttention(nn.Module):
    """自适应物理约束注意力（学习关键波段）"""

    def __init__(self, embed_dim, num_heads, sequence_length, k=3):
        super().__init__()
        self.sequence_length = sequence_length
        self.k = k  # 要关注的关键波段数量
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        # 可学习参数，表示每个波段的重要性
        self.band_importance = nn.Parameter(torch.zeros(1, sequence_length))
        nn.init.normal_(self.band_importance, mean=0, std=0.02)

    def forward(self, x):
        B, L, D = x.shape
        # 计算重要性分数：使用softmax获取概率分布
        importance_scores = F.softmax(self.band_importance, dim=-1)  # (1, L)
        # 选择top-k个波段索引
        topk_scores, topk_indices = torch.topk(importance_scores, self.k, dim=-1)  # (1, k)
        # 创建注意力掩码：只允许关注top-k波段
        mask = torch.zeros(L, L, dtype=torch.bool, device=x.device)
        for idx in topk_indices.squeeze(0).tolist():
            if idx < L:
                mask[:, idx] = True

        # 计算注意力
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)


class ParallelAttentionLayer(nn.Module):
    """并行注意力层（局部稀疏+自适应物理约束）"""

    def __init__(self, embed_dim, num_heads, window_size, sequence_length, k=3):
        super().__init__()
        self.local_attn = LocalSparseAttention(embed_dim, num_heads, window_size)
        self.physic_attn = AdaptivePhysicallyConstrainedAttention(embed_dim, num_heads, sequence_length, k)

    def forward(self, x):
        local_out = self.local_attn(x)
        physic_out = self.physic_attn(x)
        return local_out + physic_out


class GlobalSparseAttention(nn.Module):
    """全局稀疏注意力（较大窗口）"""

    def __init__(self, embed_dim, num_heads, window_size=20):
        super().__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, L, D = x.shape
        # 创建全局注意力掩码（比局部注意力更大的窗口）
        mask = torch.ones(L, L, dtype=torch.bool, device=x.device)
        for i in range(L):
            start = max(0, i - self.window_size // 2)
            end = min(L, i + self.window_size // 2)
            mask[i, :start] = False
            mask[i, end:] = False

        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)


class CNNBranch(nn.Module):
    """CNN分支（保持不变）"""

    def __init__(self, input_size):
        super().__init__()

        # 分支1：核3 + dilation=1
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(Config.leaky_relu_slope),
            nn.Conv1d(64, 64, kernel_size=3, groups=64, padding=1, dilation=1),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(Config.leaky_relu_slope)
        )

        # 分支2：核5 + dilation=3
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, dilation=3, padding=6),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(Config.leaky_relu_slope),
            nn.Conv1d(64, 64, kernel_size=5, groups=64, padding=6, dilation=3),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(Config.leaky_relu_slope)
        )

        # 分支3：核7 + dilation=5
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, dilation=5, padding=15),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(Config.leaky_relu_slope),
            nn.Conv1d(64, 64, kernel_size=7, groups=64, padding=15, dilation=5),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(Config.leaky_relu_slope)
        )

        # SE注意力模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(192, 12),
            nn.LeakyReLU(Config.leaky_relu_slope),
            nn.Linear(12, 192),
            nn.Sigmoid(),
            nn.Unflatten(1, (192, 1))
        )

        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        concat_feat = torch.cat([b1, b2, b3], dim=1)
        weights = self.se(concat_feat)
        weighted_feat = concat_feat * weights
        pooled_feat = self.pool(weighted_feat)

        return concat_feat, pooled_feat  # 返回两个特征


class NewTransformerBranch(nn.Module):
    """新的Transformer分支（基于SpectralTransformer修改）"""

    def __init__(self, input_size):
        super().__init__()

        # 1. 数据嵌入层
        self.embedding = nn.Linear(1, Config.embed_dim)

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(Config.embed_dim)

        # 3. 1D-CNN特征提取层
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=Config.embed_dim,
                out_channels=Config.cnn_filters,
                kernel_size=Config.cnn_kernel_size,
                padding=Config.cnn_kernel_size // 2
            ),
            nn.BatchNorm1d(Config.cnn_filters),
            nn.GELU(),
            nn.Conv1d(
                in_channels=Config.cnn_filters,
                out_channels=Config.embed_dim,
                kernel_size=Config.cnn_kernel_size,
                padding=Config.cnn_kernel_size // 2
            ),
            nn.BatchNorm1d(Config.embed_dim),
            nn.GELU()
        )

        # 4. 注意力层
        # 第一层：标准Transformer层
        self.stage1 = StandardTransformerLayer(
            Config.embed_dim,
            Config.num_heads,
            Config.ff_dim,
            Config.dropout_rate
        )

        # 第二层：并行注意力层
        self.stage2 = ParallelAttentionLayer(
            Config.embed_dim,
            Config.num_heads,
            Config.local_window_size,
            input_size,  # 序列长度
            Config.k_bands
        )

        # 第三层：全局稀疏注意力
        self.stage3 = GlobalSparseAttention(
            Config.embed_dim,
            Config.num_heads,
            window_size=Config.global_window_size
        )

    def forward(self, x):
        # 输入形状: (B, 1, L)
        B = x.size(0)
        L = x.size(2)

        # 嵌入层
        x = x.permute(0, 2, 1)  # (B, L, 1)
        x = self.embedding(x)  # (B, L, D)
        x = self.pos_encoder(x)

        # 1D-CNN特征提取
        x = x.permute(0, 2, 1)  # (B, D, L)
        x = self.cnn_feature_extractor(x)  # (B, D, L)
        x = x.permute(0, 2, 1)  # (B, L, D)

        # 阶段1：标准Transformer
        x1 = self.stage1(x)

        # 阶段2：并行注意力
        x2 = self.stage2(x1)

        # 阶段3：全局注意力
        x3 = self.stage3(x2)

        return x1, x2, x3  # 返回多阶段特征


class ShallowFusionModule(nn.Module):
    """浅层融合模块（门控融合）"""

    def __init__(self, cnn_channels, tr_dim, out_dim=128):
        super().__init__()
        # 特征对齐
        self.cnn_proj = nn.Conv1d(cnn_channels, tr_dim, kernel_size=1)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(tr_dim * 2, tr_dim),
            nn.Sigmoid()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(tr_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, cnn_feat, tr_feat):
        # CNN特征投影 [B, C, L] -> [B, D, L] -> [B, L, D]
        cnn_proj = self.cnn_proj(cnn_feat).permute(0, 2, 1)

        # 拼接特征 [B, L, D+D]
        combined = torch.cat([cnn_proj, tr_feat], dim=-1)

        # 生成门控权重 [B, L, D]
        gate_weights = self.gate(combined)

        # 门控融合
        fused = gate_weights * cnn_proj + (1 - gate_weights) * tr_feat

        # 非线性变换
        return self.fusion(fused)


class CrossAttention(nn.Module):
    """交叉注意力模块"""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, query, key_value):
        attn_output, _ = self.attention(
            query=query,
            key=key_value,
            value=key_value
        )
        return attn_output


class MiddleFusionModule(nn.Module):
    """中层融合模块（双路径注意力）"""

    def __init__(self, cnn_channels, tr_dim, out_dim=128, num_heads=8):
        super().__init__()
        # 特征对齐
        self.cnn_proj = nn.Conv1d(cnn_channels, tr_dim, kernel_size=1)

        # 交叉注意力
        self.cross_attn = CrossAttention(tr_dim, num_heads)

        # 全局交叉注意力
        self.global_cross = CrossAttention(tr_dim, num_heads)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(tr_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(Config.leaky_relu_slope)
        )

    def forward(self, cnn_feat, tr_feat):
        # CNN特征投影 [B, C, L] -> [B, D, L] -> [B, L, D]
        cnn_proj = self.cnn_proj(cnn_feat).permute(0, 2, 1)

        # 局部交叉注意力路径
        local_fusion = self.cross_attn(cnn_proj, tr_feat)

        # 全局交叉注意力路径
        global_vec = torch.mean(tr_feat, dim=1, keepdim=True)  # [B, 1, D]
        global_fusion = self.global_cross(cnn_proj, global_vec)

        # 双路径融合
        combined = torch.cat([local_fusion, global_fusion], dim=-1)  # [B, L, tr_dim*2]

        # 应用融合层
        return self.fusion(combined)  # [B, L, out_dim]


class FeatureConcatenationFusionWithDual(nn.Module):
    """带双重融合的特征拼接模型"""

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.input_size = input_size

        # 初始化两个分支
        self.cnn_branch = CNNBranch(input_size)
        self.transformer_branch = NewTransformerBranch(input_size)

        # 浅层和中层融合模块
        self.shallow_fusion = ShallowFusionModule(
            cnn_channels=192,
            tr_dim=Config.embed_dim,
            out_dim=128
        )
        self.middle_fusion = MiddleFusionModule(
            cnn_channels=192,
            tr_dim=Config.embed_dim,
            out_dim=128,
            num_heads=Config.num_heads
        )

        # 计算特征维度
        self.cnn_feature_dim = 192 * (input_size // 2)
        self.transformer_feature_dim = Config.embed_dim
        self.shallow_feature_dim = 128
        self.middle_feature_dim = 128
        self.total_feature_dim = self.cnn_feature_dim + self.transformer_feature_dim + self.shallow_feature_dim + self.middle_feature_dim

        # 特征融合层（带双重残差连接）
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.total_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(Config.leaky_relu_slope),
            nn.Dropout(Config.fusion_dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(Config.leaky_relu_slope),
            nn.Dropout(Config.fusion_dropout)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(Config.leaky_relu_slope),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 获取CNN分支的特征
        cnn_concat, cnn_se = self.cnn_branch(x)
        cnn_features = cnn_se.view(cnn_se.size(0), -1)  # [B, 192*(L//2)]

        # 获取Transformer分支的特征
        tr_stage1, tr_stage2, tr_stage3 = self.transformer_branch(x)
        transformer_features = tr_stage3.mean(dim=1)  # [B, D]

        # 浅层融合
        shallow_fused = self.shallow_fusion(cnn_concat, tr_stage1)
        shallow_vector = shallow_fused.mean(dim=1)  # [B, 128]

        # 中层融合
        middle_fused = self.middle_fusion(cnn_se, tr_stage2)
        middle_vector = middle_fused.mean(dim=1)  # [B, 128]

        # 最终特征拼接
        final_features = torch.cat([cnn_features, transformer_features, shallow_vector, middle_vector], dim=1)

        # 特征融合
        fused_features = self.fusion_layer(final_features)

        # 分类
        return self.classifier(fused_features)

def elastic_net_regularization(model, l1_lambda, l2_lambda):
    """计算Elastic Net正则化项"""
    l1_reg = sum(p.abs().sum() for p in model.parameters())
    l2_reg = sum(p.pow(2).sum() for p in model.parameters())
    return l1_lambda * l1_reg + l2_lambda * l2_reg


def calculate_flops(model, input_size, device):
    """计算模型的FLOPs"""
    model = model.to(device)
    model.eval()

    def count_conv1d_flops(module, input, output):
        if isinstance(module, nn.Conv1d):
            batch_size, in_channels, in_length = input[0].shape
            out_channels, out_length = output.shape[1:]
            kernel_ops = module.kernel_size[0] * in_channels
            flops = out_length * out_channels * kernel_ops
            if module.bias is not None:
                flops += out_channels * out_length
            module.flops = flops * 2

    def count_linear_flops(module, input, output):
        if isinstance(module, nn.Linear):
            flops = 2 * module.in_features * module.out_features
            if module.bias is not None:
                flops += module.out_features
            module.flops = flops

    def count_bn1d_flops(module, input, output):
        if isinstance(module, nn.BatchNorm1d):
            module.flops = 4 * output.numel()

    # 注册hook
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d)):
            hook = module.register_forward_hook(
                count_conv1d_flops if isinstance(module, nn.Conv1d) else
                count_linear_flops if isinstance(module, nn.Linear) else
                count_bn1d_flops
            )
            hooks.append(hook)

    # 前向传播一次以计算FLOPs
    dummy_input = torch.randn(1, 1, input_size).to(device)
    with torch.no_grad():
        _ = model(dummy_input)

    # 移除hook
    for hook in hooks:
        hook.remove()

    # 汇总FLOPs
    total_flops = 0
    for module in model.modules():
        if hasattr(module, 'flops'):
            total_flops += module.flops

    return total_flops


def measure_model_performance(model, input_size, device):
    """测量模型性能指标"""
    model = model.to(device)
    model.eval()

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算FLOPs
    total_flops = calculate_flops(model, input_size, device)

    # 测量推理时间
    dummy_input = torch.randn(1, 1, input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 正式测量
    num_runs = 100
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end_time = time.time()

    inference_time_per_sample = (end_time - start_time) / num_runs
    fps = 1.0 / inference_time_per_sample

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_params_M': total_params / 1e6,
        'flops': total_flops,
        'flops_G': total_flops / 1e9,
        'inference_time_ms': inference_time_per_sample * 1000,
        'fps': fps
    }


def train_model(model, train_loader, test_loader, criterion, optimizer,
                config, epochs=50, device='cpu', results_dir=None,
                progress_callback=None):
    """
    训练模型并记录训练过程
    新增参数：
        config: Config 实例，用于获取正则化系数
        progress_callback: 每个 epoch 结束后调用的函数，接收当前 epoch 编号（从1开始）
    """
    model = model.to(device)
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    best_test_acc = 0
    epoch_times = []

    log_file = os.path.join(results_dir, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write("Epoch\tTrain Loss\tTrain Acc\tTest Loss\tTest Acc\tEpoch Time\n")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (spectra, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
            spectra, labels = spectra.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(spectra)
            loss = criterion(outputs, labels)
            # 使用传入的 config 获取正则化系数
            regularization = elastic_net_regularization(model, config.l1_lambda, config.weight_decay)
            total_loss = loss + regularization
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试阶段
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for spectra, labels in test_loader:
                spectra, labels = spectra.to(device), labels.to(device)
                outputs = model(spectra)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))

        # 打印日志（仍使用 print，但可在 run_experiment 中重定向）
        print(f'\nEpoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
        print(f'Epoch Time: {epoch_time:.2f}s')

        with open(log_file, 'a') as f:
            f.write(
                f"{epoch + 1}\t{train_loss:.4f}\t{train_acc:.2f}\t{test_loss:.4f}\t{test_acc:.2f}\t{epoch_time:.2f}\n"
            )

        # 发送进度信号
        if progress_callback:
            progress_callback(epoch + 1)

    return train_losses, train_accs, test_losses, test_accs, epoch_times

def evaluate_model(model, loader, device, num_classes, idx_to_label, dataset_name="", results_dir=None):
    """评估模型性能"""
    model = model.to(device)
    model.eval()
    all_preds_idx = []
    all_labels_idx = []

    with torch.no_grad():
        for spectra, labels in loader:
            spectra, labels = spectra.to(device), labels.to(device)
            outputs = model(spectra)
            _, predicted = outputs.max(1)
            all_preds_idx.extend(predicted.cpu().numpy())
            all_labels_idx.extend(labels.cpu().numpy())

    # 映射回原始标签
    all_preds = [idx_to_label[i] for i in all_preds_idx]
    all_labels = [idx_to_label[i] for i in all_labels_idx]

    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    report_dict = classification_report(all_labels, all_preds, output_dict=True)

    macro_precision = report_dict['macro avg']['precision']
    macro_recall = report_dict['macro avg']['recall']
    macro_f1 = report_dict['macro avg']['f1-score']

    weighted_precision = report_dict['weighted avg']['precision']
    weighted_recall = report_dict['weighted avg']['recall']
    weighted_f1 = report_dict['weighted avg']['f1-score']

    print(f"\n{dataset_name}评估结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"Kappa系数: {kappa:.4f}")

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds, labels=list(idx_to_label.values()))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(idx_to_label.values()),
                yticklabels=list(idx_to_label.values()))
    plt.title(f'{dataset_name}混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(os.path.join(results_dir, f'{dataset_name}_confusion_matrix.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_labels_idx': all_labels_idx,
        'all_preds_idx': all_preds_idx
    }


def evaluate_all_sets(model, train_loader, test_loader, device, num_classes, idx_to_label, results_dir):
    print("\n" + "=" * 60)
    print("模型整体评估")
    print("=" * 60)

    train_results = evaluate_model(
        model, train_loader, device, num_classes, idx_to_label, "训练集", results_dir
    )

    print("\n测试集评估:")
    test_results = evaluate_model(
        model, test_loader, device, num_classes, idx_to_label, "测试集", results_dir
    )

    print("\n" + "=" * 60)
    print("模型整体性能总结")
    print("=" * 60)
    print(f"训练集准确率: {train_results['accuracy']:.4f}")
    print(f"测试集准确率: {test_results['accuracy']:.4f}")
    print(f"训练集Kappa系数: {train_results['kappa']:.4f}")
    print(f"测试集Kappa系数: {test_results['kappa']:.4f}")

    return {
        'train': train_results,
        'test': test_results
    }

def plot_training_curves(train_losses, train_accs, test_losses, test_accs, results_dir=None):
    """绘制训练和测试曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.plot(test_losses, 'r-', label='测试损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.title('损失曲线')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='训练准确率')
    plt.plot(test_accs, 'r-', label='测试准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.title('准确率曲线')
    plt.grid(True)

    plt.tight_layout()

    # 保存训练曲线图像
    if results_dir:
        plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')

    plt.close()

def run_experiment(config, log_callback=None, progress_callback=None):
    """
    执行完整实验，返回结果字典
    参数：
        config: Config 实例，包含所有配置
        log_callback: 用于输出日志的函数，接收字符串
        progress_callback: 用于报告进度的函数，接收当前 epoch 编号（从1开始）
    """
    # 备份原始的 print 函数，用于恢复
    original_print = builtins.print

    # 如果提供了 log_callback，则重定向 print
    if log_callback:
        def custom_print(*args, **kwargs):
            msg = ' '.join(str(arg) for arg in args)
            log_callback(msg + '\n')
            original_print(*args, **kwargs)
        builtins.print = custom_print

    try:
        # 将传入的 config 实例的属性更新到 Config 类，使模型创建时使用新值
        for key, value in vars(config).items():
            if not key.startswith('__'):
                setattr(Config, key, value)

        # 创建结果目录（使用 config 中的 results_dir 作为基目录）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(config.results_dir, f"experiment_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # 保存配置到文件
        config_file = os.path.join(results_dir, 'config.json')
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__') and not callable(v)}
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)

        print("\n=== 模型配置 ===")
        print(f"数据文件: {config.file_path}")
        print(f"数据表格: {config.sheet_name}")
        print(f"训练轮数: {config.epochs}")
        print(f"批次大小: {config.batch_size}")
        print(f"学习率: {config.learning_rate}")
        print(f"L2正则化系数: {config.weight_decay}")
        print(f"L1正则化系数: {config.l1_lambda}")
        print(f"数据集划分: 训练集 {(1 - config.test_size) * 100:.1f}%, 测试集 {config.test_size * 100:.1f}%")
        print(f"结果保存路径: {results_dir}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n使用设备: {device}")

        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        # 加载数据
        data, labels, wavelengths = load_data(config.file_path, config.sheet_name)

        # 创建标签映射
        unique_labels = np.unique(labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        num_classes = len(unique_labels)

        # 保存标签映射
        label_mapping_file = os.path.join(results_dir, 'label_mapping.json')
        with open(label_mapping_file, 'w') as f:
            json.dump(label_to_idx, f, indent=4)

        print("\n标签映射关系:")
        for label, idx in label_to_idx.items():
            print(f"原始标签: {label} -> 索引: {idx}")

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels,
            test_size=config.test_size,
            stratify=labels,
            random_state=config.random_seed
        )

        train_dataset = SpectralDataset(X_train, y_train, label_to_idx)
        test_dataset = SpectralDataset(X_test, y_test, label_to_idx)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # 创建模型
        model = FeatureConcatenationFusionWithDual(input_size=data.shape[1], num_classes=num_classes)

        # 测量模型性能
        performance_metrics = measure_model_performance(model, data.shape[1], device)
        print(f"模型参数总量: {performance_metrics['total_params']:,}")
        print(f"可训练参数: {performance_metrics['trainable_params']:,}")
        print(f"参数总量 (M): {performance_metrics['total_params_M']:.2f}M")
        print(f"FLOPs: {performance_metrics['flops']:,}")
        print(f"FLOPs (G): {performance_metrics['flops_G']:.4f}G")
        print(f"单样本推理时间: {performance_metrics['inference_time_ms']:.2f}ms")
        print(f"推理速度 (FPS): {performance_metrics['fps']:.2f}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        # 训练
        train_losses, train_accs, test_losses, test_accs, epoch_times = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            config=config, epochs=config.epochs, device=device, results_dir=results_dir,
            progress_callback=progress_callback
        )

        avg_epoch_time = np.mean(epoch_times)

        # 绘制训练曲线
        plot_training_curves(train_losses, train_accs, test_losses, test_accs, results_dir)

        # 加载最佳模型并评估
        model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
        model = model.to(device)

        evaluation_results = evaluate_all_sets(
            model, train_loader, test_loader, device, num_classes, idx_to_label, results_dir
        )

        # 收集结果
        results = {
            'model_performance': performance_metrics,
            'training_time': {
                'total_training_time_seconds': sum(epoch_times),
                'average_epoch_time_seconds': avg_epoch_time,
                'epoch_times': [float(t) for t in epoch_times]
            },
            'train_accuracy': evaluation_results['train']['accuracy'],
            'test_accuracy': evaluation_results['test']['accuracy'],
            'train_kappa': evaluation_results['train']['kappa'],
            'test_kappa': evaluation_results['test']['kappa'],
            'train_macro_precision': evaluation_results['train']['macro_precision'],
            'test_macro_precision': evaluation_results['test']['macro_precision'],
            'train_macro_recall': evaluation_results['train']['macro_recall'],
            'test_macro_recall': evaluation_results['test']['macro_recall'],
            'train_macro_f1': evaluation_results['train']['macro_f1'],
            'test_macro_f1': evaluation_results['test']['macro_f1'],
            'train_weighted_precision': evaluation_results['train']['weighted_precision'],
            'test_weighted_precision': evaluation_results['test']['weighted_precision'],
            'train_weighted_recall': evaluation_results['train']['weighted_recall'],
            'test_weighted_recall': evaluation_results['test']['weighted_recall'],
            'train_weighted_f1': evaluation_results['train']['weighted_f1'],
            'test_weighted_f1': evaluation_results['test']['weighted_f1'],
            'train_losses': [float(loss) for loss in train_losses],
            'train_accuracies': [float(acc) for acc in train_accs],
            'test_losses': [float(loss) for loss in test_losses],
            'test_accuracies': [float(acc) for acc in test_accs],
            'wavelengths': wavelengths.tolist(),
            'label_mapping': label_to_idx
        }

        # 保存完整结果
        torch.save({
            'model_state_dict': model.state_dict(),
            'results': results,
            'config': config_dict
        }, os.path.join(results_dir, 'feature_concat_results.pth'))

        with open(os.path.join(results_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print("\n训练和评估完成！所有结果已保存到:", results_dir)

        return results

    finally:
        # 恢复原始的 print 函数
        builtins.print = original_print

# # main 函数
# def main():
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     results_dir = os.path.join(Config.results_dir, f"experiment_{timestamp}")
#     os.makedirs(results_dir, exist_ok=True)

#     config_file = os.path.join(results_dir, 'config.json')
#     config_dict = {k: v for k, v in vars(Config).items() if not k.startswith('__') and not callable(v)}
#     with open(config_file, 'w') as f:
#         json.dump(config_dict, f, indent=4)

#     print("\n=== 模型配置 ===")
#     print(f"数据文件: {Config.file_path}")
#     print(f"数据表格: {Config.sheet_name}")
#     print(f"训练轮数: {Config.epochs}")
#     print(f"批次大小: {Config.batch_size}")
#     print(f"学习率: {Config.learning_rate}")
#     print(f"L2正则化系数: {Config.weight_decay}")
#     print(f"L1正则化系数: {Config.l1_lambda}")
#     print(f"数据集划分: 训练集 {(1 - Config.test_size) * 100:.1f}%, 测试集 {Config.test_size * 100:.1f}%")
#     print(f"结果保存路径: {results_dir}")

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"\n使用设备: {device}")

#     torch.manual_seed(Config.random_seed)
#     np.random.seed(Config.random_seed)

#     data, labels, wavelengths = load_data(Config.file_path, Config.sheet_name)

#     # 创建标签映射
#     unique_labels = np.unique(labels)
#     label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
#     idx_to_label = {idx: label for label, idx in label_to_idx.items()}
#     num_classes = len(unique_labels)

#     # 保存标签映射
#     label_mapping_file = os.path.join(results_dir, 'label_mapping.json')
#     with open(label_mapping_file, 'w') as f:
#         json.dump(label_to_idx, f, indent=4)

#     print("\n标签映射关系:")
#     for label, idx in label_to_idx.items():
#         print(f"原始标签: {label} -> 索引: {idx}")

#     X_train, X_test, y_train, y_test = train_test_split(
#         data, labels,
#         test_size=Config.test_size,
#         stratify=labels,
#         random_state=Config.random_seed
#     )

#     train_dataset = SpectralDataset(X_train, y_train, label_to_idx)
#     test_dataset = SpectralDataset(X_test, y_test, label_to_idx)
#     train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

#     model = FeatureConcatenationFusionWithDual(input_size=data.shape[1], num_classes=num_classes)

#     performance_metrics = measure_model_performance(model, data.shape[1], device)
#     print(f"模型参数总量: {performance_metrics['total_params']:,}")
#     print(f"可训练参数: {performance_metrics['trainable_params']:,}")
#     print(f"参数总量 (M): {performance_metrics['total_params_M']:.2f}M")
#     print(f"FLOPs: {performance_metrics['flops']:,}")
#     print(f"FLOPs (G): {performance_metrics['flops_G']:.4f}G")
#     print(f"单样本推理时间: {performance_metrics['inference_time_ms']:.2f}ms")
#     print(f"推理速度 (FPS): {performance_metrics['fps']:.2f}")

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

#     train_losses, train_accs, test_losses, test_accs, epoch_times = train_model(
#         model, train_loader, test_loader, criterion, optimizer,
#         epochs=Config.epochs, device=device, results_dir=results_dir
#     )

#     avg_epoch_time = np.mean(epoch_times)

#     plot_training_curves(train_losses, train_accs, test_losses, test_accs, results_dir)

#     model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
#     model = model.to(device)

#     evaluation_results = evaluate_all_sets(
#         model, train_loader, test_loader, device, num_classes, idx_to_label, results_dir
#     )

#     # 保存结果
#     results = {
#         'model_performance': performance_metrics,
#         'training_time': {
#             'total_training_time_seconds': sum(epoch_times),
#             'average_epoch_time_seconds': avg_epoch_time,
#             'epoch_times': [float(t) for t in epoch_times]
#         },
#         'train_accuracy': evaluation_results['train']['accuracy'],
#         'test_accuracy': evaluation_results['test']['accuracy'],
#         'train_kappa': evaluation_results['train']['kappa'],
#         'test_kappa': evaluation_results['test']['kappa'],
#         'train_macro_precision': evaluation_results['train']['macro_precision'],
#         'test_macro_precision': evaluation_results['test']['macro_precision'],
#         'train_macro_recall': evaluation_results['train']['macro_recall'],
#         'test_macro_recall': evaluation_results['test']['macro_recall'],
#         'train_macro_f1': evaluation_results['train']['macro_f1'],
#         'test_macro_f1': evaluation_results['test']['macro_f1'],
#         'train_weighted_precision': evaluation_results['train']['weighted_precision'],
#         'test_weighted_precision': evaluation_results['test']['weighted_precision'],
#         'train_weighted_recall': evaluation_results['train']['weighted_recall'],
#         'test_weighted_recall': evaluation_results['test']['weighted_recall'],
#         'train_weighted_f1': evaluation_results['train']['weighted_f1'],
#         'test_weighted_f1': evaluation_results['test']['weighted_f1'],
#         'train_losses': [float(loss) for loss in train_losses],
#         'train_accuracies': [float(acc) for acc in train_accs],
#         'test_losses': [float(loss) for loss in test_losses],
#         'test_accuracies': [float(acc) for acc in test_accs],
#         'wavelengths': wavelengths.tolist(),
#         'label_mapping': label_to_idx
#     }

#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'results': results,
#         'config': config_dict
#     }, os.path.join(results_dir, 'feature_concat_results.pth'))

#     with open(os.path.join(results_dir, 'results.json'), 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=4, ensure_ascii=False)

#     print("\n训练和评估完成！所有结果已保存到:", results_dir)



# if __name__ == '__main__':
#     main()

