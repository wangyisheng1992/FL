import os
import re
import copy
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

# 禁用不必要的警告
import warnings
warnings.filterwarnings('ignore')

# 配置信息
CONFIG = {
    "daic_woz_root": r"F:\DAIC-WOZ",  # DAIC-WOZ数据集根目录
    "cache_dir": "./cache",  # 缓存目录
    "batch_size": 16,  # 批次大小

    "epochs": 50,  # 训练轮数
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 计算设备
    "phq8_threshold": 10,  # PHQ-8抑郁阈值
    "max_length": 256,
    "id_column": "Participant_ID",  # 数据框中参与者ID列名
    "phq8_column": "PHQ8_Score",  # 数据框中PHQ-8列名
    "model_save_dir": "./saved_models",  # 模型保存目录
    "best_model": "best_model.pth",
    "last_model": "last_model.pth",
    "best_threshold": "best_threshold",

    "au_feature_dim": 32,      # 17个AU的强度+出现频率
    "eye_feature_dim": 2,      # 眨眼率/注视比等
    "pose_feature_dim": 6,     # 头部角度/运动幅度
    "video_segments": 16,  # 时间步/片段数量
    "video_feature_per_segment": 8, # 每个片段的特征维度
    "feature_scaling": True,
    # 降低初始学习率
    "learning_rate": 1e-4,

    # 使用混合精度
    "use_amp": False,
    "visual_processor": {
        "layer_sizes": [8, 64, 128, 256],  # 明确的特征学习路径
        "use_residual": True,
        "activation": "relu"
    },


    "train_files": [  # 训练集CSV文件列表
        "F:/DAIC-WOZ/dev_split_Depression_AVEC2017.csv",
        "F:/DAIC-WOZ/train_Depression_AVEC2017.csv"
    ],
    "test_file": "F:/DAIC-WOZ/full_test_split.csv",  # 测试集CSV文件
    "min_depression_feature": 0.4,  # 最小抑郁特征强度
    "dsm_weights": {  # DSM-5各维度权重
        "core_emotion": 1.5,
        "interest_loss": 1.2,
        "sleep_disorder": 1.0,
        "energy_change": 1.0,
        "cognitive_function": 1.2,
        "self_evaluation": 1.5,
        "suicide_risk": 2.0,
        "somatic_symptoms": 1.0,
        "social_function": 1.2
    },
    "transformer_weights": "depression_transformer.pth",  # Transformer模型权重路径
    "facial_feature_dim": 17,  # 面部特征维度
    "video_feature_dim": 604,  # 视频特征维度 (符合MIL模型要求)
    "audio_feature_dim": 768,  # 音频特征维度 (符合MIL模型要求)
    "text_feature_dim": 768,  # 文本特征维度
    "video_segments": 16,  # 新增键
    "blink_threshold": 15,  # 正常眨眼频率阈值(次/分钟)
    "gaze_threshold": 0.25,  # 目光接触减少阈值
    "head_movement_threshold": 0.15,  # 头部运动减少阈值
    "head_down_threshold": -0.3,  # 低头姿势阈值
    "facial_expression_threshold": 0.5,  # 面部表情减少阈值
    "clip_value": 0.5,  # 梯度裁剪阈值
    "init_scale": 0.02,  # 参数初始化范围
    "patience": 5,  # 早停耐心值
    "early_stopping": True,  # 是否启用早停
    "threshold_search": True,  # 是否启用最佳阈值搜索
    "bag_size": 3,  # 多实例学习的包大小
    "model_save_dir": "models",
    "use_sigmoid": True          # 是否使用Sigmoid
}

# 在CONFIG中新增特征维度
CONFIG.update({
    "au_feature_dim": 32,  # 17个AU的强度+出现频率
    "eye_feature_dim": 2,  # 眨眼率/注视比等
    "pose_feature_dim": 6,  # 头部角度/运动幅度
})

os.makedirs(CONFIG["model_save_dir"], exist_ok=True
# 创建缓存目录
if not os.path.exists(CONFIG["cache_dir"]):
    os.makedirs(CONFIG["cache_dir"]))

def read_video_txt(file_path):
        return pd.DataFrame()
    
# 辅助函数: 重新排列张量维度
def rearrange(tensor, pattern, **axes_lengths):
    """扩展张量重组函数以支持多头注意力需要的模式"""
    # 已有模式保持不变
    if pattern == 'b t1 (t2 w) -> (b t1) t2 w':
        b, t1, rest = tensor.shape
        t2 = axes_lengths['t2']
        w = rest // t2
        return tensor.reshape(b * t1, t2, w)

    elif pattern == '(b t) c -> b t c':
        bt, c = tensor.shape
        t = axes_lengths['t']
        b = bt // t
        return tensor.reshape(b, t, c)

    # 新增以下两种模式以支持多头注意力
    # 模式: b n (h d) -> b h n d
    elif pattern == 'b n (h d) -> b h n d':
        b, n, hid_dim = tensor.shape
        h = axes_lengths['h']
        d = hid_dim // h
        return tensor.reshape(b, n, h, d).permute(0, 2, 1, 3)

    # 模式: b h n d -> b n (h d)
    elif pattern == 'b h n d -> b n (h d)':
        b, h, n, d = tensor.shape
        return tensor.permute(0, 2, 1, 3).reshape(b, n, h * d)

    else:
        raise ValueError(f"不支持的pattern: {pattern}")


# 抑郁症专用Transformer编码器
class DepressionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 轻量级配置（2层Transformer）
        self.config = {
            "hidden_size": 768,
            "num_hidden_layers": 2,
            "num_attention_heads": 12,
            "intermediate_size": 512,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "layer_norm_eps": 1e-12
        }

        # 手动实现轻量级Transformer
        self.layer_norm = nn.LayerNorm(self.config["hidden_size"], eps=self.config["layer_norm_eps"])
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.config["hidden_size"],
                nhead=self.config["num_attention_heads"],
                dim_feedforward=self.config["intermediate_size"],
                dropout=self.config["hidden_dropout_prob"],
                activation=self.config["hidden_act"],
                batch_first=True
            )
            for _ in range(self.config["num_hidden_layers"])
        ])
        self.dropout = nn.Dropout(self.config["hidden_dropout_prob"])

    def forward(self, embeddings):

        """输入: (batch_size, 768) 原始特征, 输出: (batch_size, 768) 精炼后的抑郁症特征"""
        # 跳过单标记的位置编码
        embeddings = embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 仅应用层归一化和Dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # 通过Transformer层
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states[:, 0, :]

    def position_embeddings(self, position_ids):
        """生成位置嵌入"""
        # 简化版位置编码
        position_embeddings = torch.zeros_like(position_ids, dtype=torch.float)
        position = position_ids.float()

        # 使用正弦和余弦函数生成位置编码
        div_term = torch.exp(torch.arange(0, self.config["hidden_size"], 2).float() *
                             (-torch.log(torch.tensor(10000.0)) / self.config["hidden_size"]))
        position_embeddings[:, 0::2] = torch.sin(position * div_term)
        position_embeddings[:, 1::2] = torch.cos(position * div_term)

        return position_embeddings

# 从models.py导入的模型类
class video_BiLSTM_fea(nn.Module):
    """用于处理视频特征，再融合进文本模型"""

    def __init__(self):
        super().__init__()
        self.input_size = 512
        self.hidden_size = 256
        self.num_layers = 3
        self.output_size = 256
        self.dropout_rate = 0.3

        # 定义双向LSTM层
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.batch_norm = nn.BatchNorm1d(self.hidden_size * 2)

        # 定义全连接层
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        self.bn_fc = nn.BatchNorm1d(self.output_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = nn.ReLU()
        self.fc_t1 = nn.Linear(256, 256)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 批归一化
        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)

        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        out = self.bn_fc(out)
        out = self.dropout(out)
        out = self.activation(out)
        out = self.fc_t1(out)

        return out


class audio_BiLSTM_Fea(nn.Module):
    """用于处理音频特征"""

    def __init__(self):
        super().__init__()
        self.input_size = 768
        self.hidden_size = 256
        self.num_layers = 3
        self.output_size = 256
        self.dropout_rate = 0.2

        # 定义双向LSTM层
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.batch_norm = nn.BatchNorm1d(self.hidden_size * 2)

        # 定义全连接层
        self.fc = nn.Linear(self.hidden_size * 2, self.output_size)
        self.bn_fc = nn.BatchNorm1d(self.output_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.activation = nn.ReLU()
        self.fc_t1 = nn.Linear(256, 256)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))

        # 批归一化
        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)

        # 解码最后一个时间步的隐藏状态
        out = self.fc(out[:, -1, :])
        out = self.bn_fc(out)
        out = self.dropout(out)
        out = self.activation(out)
        out = self.fc_t1(out)

        return out

class MIL_Text_Video_Audio_M3(nn.Module):
    """多模态融合模型 (文本+视频+音频)"""

    def __init__(self):
        super().__init__()
        self.video_bilstm = video_BiLSTM_fea()
        self.audio_bilstm = audio_BiLSTM_Fea()

        # 文本处理参数
        self.input_size = 768
        self.hidden_size = 768
        self.num_layers = 3
        self.output_size = 768
        self.bag_size = CONFIG["bag_size"]
        self.num_classes = 2
        self.dropout_rate = 0.3

        # 注意力机制参数
        self.heads = 4
        self.dim_head = 768 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(768, (self.dim_head * self.heads) * 3, bias=False)
        self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)

        # 文本处理LSTM
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True
        )
        self.batch_norm = nn.BatchNorm1d(self.hidden_size)

        # 文本特征提取层
        self.text_fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # 多模态融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 * 3, 128),  # 文本+视频+音频
            nn.BatchNorm1d(128),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

        # 初始化模型权重
        self._initialize_weights()

    def _initialize_weights(self):
        """改进的权重初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # 避免全零偏置
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.01)  # 偏置非零
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.05)
            for lstm_layer in [self.video_bilstm, self.audio_bilstm]:
                # 遗忘门偏置初始化为1（标准做法）
                for name, param in lstm_layer.named_parameters():
                    if 'bias' in name:
                        n = param.size(0)
                        param.data.fill_(0.01)
                        # 遗忘门偏置更大
                        param.data[n // 4:n // 2].fill_(1.0)
        # 特别注意：视频和音频BiLSTM需要单独初始化
        # self._initialize_submodule(self.video_bilstm)
        # self._initialize_submodule(self.audio_bilstm)

        # 注意力层特殊初始化（使用kaiming正态分布）
        # nn.init.kaiming_normal_(self.to_qkv.weight, mode='fan_in')

        print("✅ 模型权重初始化完成")

    def _initialize_submodule(self, module):
        """递归初始化子模块权重"""
        for name, param in module.named_parameters():
            if 'bias' in name:
                # 偏置项初始化为0.01
                nn.init.constant_(param, 0.01)
            elif 'weight' in name:
                # 检查维度
                if param.dim() >= 2:  # 只对≥2维的张量应用Xavier初始化
                    if 'lstm' in name.lower():
                        if 'hh' in name:
                            nn.init.orthogonal_(param)
                        else:
                            nn.init.xavier_uniform_(param)
                    else:
                        nn.init.xavier_uniform_(param)
                elif param.dim() == 1:
                    # 对1维权重特殊处理（如BatchNorm的weight）
                    nn.init.normal_(param, mean=1.0, std=0.02)

    def mil_aggregation(self, text_emb):
        """
        多实例注意力聚合
        Args:
            text_emb: [batch, bag_size, 768]
        Returns:
            aggregated: [batch, 768] 聚合后的文本表示
            attn_weights: [batch, bag_size] 注意力权重
        """
        # 1. 多头自注意力
        qkv = self.to_qkv(text_emb)  # [b, n, h*d*3]
        q, k, v = rearrange(qkv, 'b n (k h d) -> k b h n d', k=3, h=self.heads)

        # 2. 注意力计算
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn_weights = torch.softmax(attn, dim=-1)  # [batch, heads, bag, bag]

        # 3. 特征聚合
        aggregated = torch.einsum('b h i j, b h j d -> b h i d', attn_weights, v)
        aggregated = rearrange(aggregated, 'b h n d -> b n (h d)')

        # 4. 取最后head的注意力作为全局权重
        final_attn = attn_weights[:, -1].mean(dim=1)  # [batch, bag]

        # 5. 位置融合
        aggregated = self.pwconv(aggregated).squeeze(1)  # [batch, hidden_dim]

        return aggregated, final_attn

    def forward(self, x_text, x_video, x_audio):
        """前向传播函数"""
        # 1. 文本特征处理
        # 调整维度: [batch, bag_size, 768]
        if x_text.dim() == 4:
            x_text = x_text.squeeze(2)  # 移除多余维度

        # 文本实例处理
        x = rearrange(x_text, 'b t (d) -> (b t) l d', l=1)  # [batch*bags, 1, 768]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x_text.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x_text.device)

        # LSTM处理
        lstm_out, _ = self.lstm(x, (h0, c0))  # [batch*bags, 1, 768]
        lstm_out = lstm_out[:, -1, :]  # 取最后一个时间步
        lstm_out = rearrange(lstm_out, '(b t) d -> b t d', t=self.bag_size)  # [batch, bags, 768]

        # 2. 多实例聚合
        text_agg, text_attn = self.mil_aggregation(lstm_out)  # [batch, 768], [batch, bags]

        # 3. 文本特征降维
        text_feat = self.text_fc(text_agg)  # [batch, 256]

        # 4. 视频特征处理
        # 确保视频输入维度正确
        if x_video.dim() > 3:
            x_video = x_video.mean(dim=2)  # [batch, seq, features]
        video_feat = self.video_bilstm(x_video)  # [batch, 256]

        # 5. 音频特征处理
        # 确保音频输入维度正确
        if x_audio.dim() > 3:
            x_audio = x_audio.squeeze(1)  # 移除多余维度
        audio_feat = self.audio_bilstm(x_audio)  # [batch, 256]

        # 6. 特征融合
        fusion_fea = torch.cat((text_feat, video_feat, audio_feat), dim=1)  # [batch, 768]

        # 7. 分类预测
        logits = self.fusion_fc(fusion_fea)

        return logits, text_attn


class EnhancedMILModel(MIL_Text_Video_Audio_M3):
    def __init__(self):
        super().__init__()

        # 文本特征处理器
        self.text_processor = nn.Linear(768, 256) if CONFIG.get("text_feature_dim", 768) else None

        # 音频特征处理器
        self.audio_processor = nn.Linear(768, 256) if CONFIG.get("audio_feature_dim", 768) else None

        # 视觉特征处理器
        self.visual_processor = nn.Sequential(
            nn.Linear(8, 256),  # 修改输入维度为实际特征维度
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),
            # 添加残差连接
            nn.Conv1d(1, 1, 1)  # 效果：增强特征交叉
        )
        # 添加时间注意力层（关键添加）
        self.temporal_attention = nn.Sequential(
            nn.Linear(256, 1),  # 输入是256维视觉特征
            nn.Softmax(dim=1)  # 在时间维度上归一化
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)  # 确保输出2个类别
        )
        # 解决父类没有 visual_processor 的问题
        self.video_feature_processor = nn.Sequential(
            nn.Linear(8, 256),  # 修改输入维度
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3)
        )

    def forward(self, x_text, x_video, x_audio):
        # 处理文本特征 - 将3D (batch, bag, features) 压缩为2D
        if x_text.dim() == 3:
            x_text = x_text.mean(dim=1)  # 对bag_size维度取平均
        text_emb = self.text_processor(x_text) if self.text_processor else x_text

        # 处理音频特征 - 同样压缩为2D
        if x_audio.dim() == 3:
            x_audio = x_audio.mean(dim=1)  # 对序列维度取平均
        audio_emb = self.audio_processor(x_audio) if self.audio_processor else x_audio

        # 视觉特征处理 (保持原逻辑)
        batch_size, seq_len, _ = x_video.shape
        x_video_flat = x_video.reshape(-1, x_video.size(-1))
        visual_feat_flat = self.video_feature_processor(x_video_flat)  # 使用新名称
        visual_feat = visual_feat_flat.view(batch_size, seq_len, -1)
        attention_weights = self.temporal_attention(visual_feat)
        visual_feat = torch.sum(visual_feat * attention_weights, dim=1)  # 2D

        # 确保所有特征都是2D
        feat_list = [text_emb, visual_feat, audio_emb]

        # 特征拼接
        multimodal_feature = torch.cat(feat_list, dim=1)  # dim=1拼接特征维度

        # 分类
        output = self.classifier(multimodal_feature)
        return output, {"video_attention": attention_weights.squeeze(-1)}

# DSM-5症状分析器
class DSM5Analyzer:
    def __init__(self):
        self.symptom_dims = {
            "core_emotion": slice(0, 128),
            "interest_loss": slice(128, 256),
            "sleep_disorder": slice(256, 320),
            "energy_change": slice(320, 384),
            "cognitive_function": slice(384, 448),
            "self_evaluation": slice(448, 512),
            "suicide_risk": slice(512, 576),
            "somatic_symptoms": slice(576, 640),
            "social_function": slice(640, 768)
        }

    def calculate_dsm_index(self, feature_vector):
        """计算DSM-5综合指数"""
        total = 0.0
        weight_sum = 0.0
        for dim, slc in self.symptom_dims.items():
            dim_score = np.mean(feature_vector[slc])
            total += dim_score * CONFIG["dsm_weights"][dim]
            weight_sum += CONFIG["dsm_weights"][dim]
        return (total / weight_sum) * 100  # 转换为百分比指数

    def calculate_dsm_index_tensor(self, feature_vector_tensor):
        """计算DSM-5综合指数 (PyTorch张量版本)"""
        total = torch.zeros(feature_vector_tensor.size(0), device=feature_vector_tensor.device)
        weight_sum = 0.0
        for dim, slc in self.symptom_dims.items():
            dim_scores = feature_vector_tensor[:, slc]
            dim_score = torch.mean(dim_scores, dim=1)
            total += dim_score * CONFIG["dsm_weights"][dim]
            weight_sum += CONFIG["dsm_weights"][dim]
        return (total / weight_sum) * 100


# 简化的DeepSeek DSM特征提取器
class DeepSeekDSM:
    def __init__(self, use_transformer=True, train_mode=False):
        self.analyzer = DSM5Analyzer()
        self.use_transformer = use_transformer

        # 初始化Transformer
        if self.use_transformer:
            self.transformer = DepressionTransformer()
            self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-4)
            if os.path.exists(CONFIG["transformer_weights"]):
                print(f"加载Transformer权重: {CONFIG['transformer_weights']}")
                self.transformer.load_state_dict(torch.load(CONFIG["transformer_weights"]))
            self.train_mode = train_mode
            if self.train_mode:
                self.transformer.train()
            else:
                self.transformer.eval()

    def get_deepseek_dsm(self, participant_id):
        """直接从缓存加载DSM特征向量"""
        cache_file = os.path.join(CONFIG["cache_dir"], f"{participant_id}_dsm_embedding.npy")

        if os.path.exists(cache_file):
            embedding = np.load(cache_file)
            # 确保特征维度正确
            if embedding.shape != (CONFIG["text_feature_dim"],):
                # 调整维度
                if embedding.shape[0] > CONFIG["text_feature_dim"]:
                    embedding = embedding[:CONFIG["text_feature_dim"]]
                else:
                    padding = np.zeros(CONFIG["text_feature_dim"] - embedding.shape[0])
                    embedding = np.concatenate([embedding, padding])
        else:
            # 如果缓存不存在，创建全零向量
            embedding = np.zeros(CONFIG["text_feature_dim"])

        # 如果启用Transformer，进一步处理特征
        if self.use_transformer and embedding is not None:
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(CONFIG["device"])
            with torch.no_grad():
                refined_embedding = self.transformer(embedding_tensor).squeeze(0).cpu().numpy()
            return refined_embedding

        return embedding


# 视频特征处理模块（符合MIL模型要求）
class VideoProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.segment_length = CONFIG["video_segments"]
        self.feature_dim = CONFIG["video_feature_per_segment"]
        print(f"视频处理器: segments={self.segment_length}, features={self.feature_dim}")
        self.segment_length = CONFIG["video_segments"]
        self.feature_dim = CONFIG["video_feature_per_segment"]
        self.segment_length = CONFIG["video_segments"]  # 使用新配置项


    def _read_feature_file(self, file_path):
        """读取特征文件"""
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                has_header = any(char.isalpha() for char in first_line)
                sep = ',' if file_path.endswith('.txt') and ',' in first_line else '\t'

            if has_header:
                df = pd.read_csv(file_path, sep=sep, header=0)
                df.columns = [col.strip().strip('"').strip("'") for col in df.columns]
            else:
                df = pd.read_csv(file_path, sep=sep, header=None)

            return df
        except Exception as e:
            print(f"读取文件失败: {file_path}, 错误: {str(e)}")
            return None

    def process_video_features(self, participant_id):
        """处理视频特征（转换为16x512格式）"""
        # 尝试多种可能的ID格式
        possible_dirs = [
            f"{participant_id}_P",
            f"{participant_id}P",
            f"P{participant_id}",
            participant_id
        ]

        clnf_path = ""
        for dir_name in possible_dirs:
            path = os.path.join(
                self.root_dir,
                dir_name,
                f"{participant_id}_CLNF_features3D.txt"
            )
            if os.path.exists(path):
                clnf_path = path
                break

        if not clnf_path:
            print(f"警告: 参与者 {participant_id} 的视频特征文件不存在")
            return np.zeros(CONFIG["video_feature_dim"], dtype=np.float32)

        try:
            df = self._read_feature_file(clnf_path)
            if df is None:
                return np.zeros(CONFIG["video_feature_dim"], dtype=np.float32)

            # 提取特征数据 (跳过前4列)
            features = df.iloc[:, 4:].values.astype(np.float32)
            num_frames, num_features = features.shape

            # 分割为16个片段
            segment_size = max(1, num_frames // self.segment_length)
            segments = []

            for i in range(self.segment_length):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, num_frames)

                if start_idx < num_frames:
                    segment = features[start_idx:end_idx]

                    # 如果特征维度不足512，填充
                    if num_features < self.feature_dim:
                        padding = np.zeros((segment.shape[0], self.feature_dim - num_features))
                        segment = np.hstack([segment, padding])
                    # 如果特征维度超过512，截断
                    elif num_features > self.feature_dim:
                        segment = segment[:, :self.feature_dim]

                    # 取片段均值作为该段特征
                    segment_mean = np.mean(segment, axis=0)
                    segments.append(segment_mean)
                else:
                    segments.append(np.zeros(self.feature_dim))

            # 转换为数组 (16, 512)
            video_features = np.array(segments)
            return video_features
        except Exception as e:
            print(f"加载参与者 {participant_id} 的视频特征失败: {str(e)}")
            return np.zeros(CONFIG["video_feature_dim"], dtype=np.float32)


# 音频特征处理模块（符合MIL模型要求）
class AudioProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.segment_length = CONFIG["video_segments"]  # 使用视频的时间片数量
        self.feature_dim = 768  # 音频特征维度直接设为768
        print(f"音频处理器: segments={self.segment_length}, features={self.feature_dim}")

    def _read_feature_file(self, file_path):
        """读取特征文件"""
        if not os.path.exists(file_path):
            return None

        try:
            df = pd.read_csv(file_path, sep=',', header=None)
            return df
        except Exception as e:
            print(f"读取文件失败: {file_path}, 错误: {str(e)}")
            return None

    def process_audio_features(self, participant_id):
        """处理音频特征（转换为16x768格式）"""
        # 尝试多种可能的ID格式
        possible_dirs = [
            f"{participant_id}_P",
            f"{participant_id}P",
            f"P{participant_id}",
            participant_id
        ]

        audio_path = ""
        for dir_name in possible_dirs:
            path = os.path.join(
                self.root_dir,
                dir_name,
                f"{participant_id}_COVAREP.csv"
            )
            if os.path.exists(path):
                audio_path = path
                break

        if not audio_path:
            print(f"警告: 参与者 {participant_id} 的音频特征文件不存在")
            return np.zeros(CONFIG["audio_feature_dim"])

        try:
            # 1. 正确读取CSV文件（指定分隔符为逗号）
            df = pd.read_csv(audio_path, sep=',', header=None)  # 关键：明确sep=','
            if df.empty:
                return np.zeros(CONFIG["audio_feature_dim"])

            # 2. 替换无效值（0值过多可能导致问题，但先保留原始值）
            features = df.values.astype(np.float32)
            features[np.isinf(features)] = 0.0  # 替换已有inf
            features[np.isnan(features)] = 0.0  # 替换已有nan

            # 3. 分割为16个片段（保持原有逻辑）
            segment_size = max(1, len(features) // self.segment_length)
            segments = []

            for i in range(self.segment_length):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, len(features))
                if start_idx >= len(features):
                    segments.append(np.zeros(self.feature_dim))
                    continue

                segment = features[start_idx:end_idx]

                # 4. 计算统计特征时添加稳定性处理
                mean = np.mean(segment, axis=0)
                std = np.std(segment, axis=0)
                # 关键：避免std=0导致后续除以0（添加微小值）
                std = np.where(std < 1e-6, 1e-6, std)  # 替换接近0的标准差

                # 5. 标准化（避免极端值）
                segment_norm = (segment - mean) / std
                segment_norm = np.clip(segment_norm, -5, 5)  # 限制在[-5,5]，避免极端值

                # 6. 计算该片段的统计特征（均值、标准差等）
                seg_mean = np.mean(segment_norm, axis=0)
                seg_std = np.std(segment_norm, axis=0)
                seg_max = np.max(segment_norm, axis=0)
                seg_min = np.min(segment_norm, axis=0)

                # 7. 拼接统计特征（确保维度正确）
                segment_feat = np.concatenate([seg_mean, seg_std, seg_max, seg_min])

                # 8. 调整维度至768（你的配置中audio_feature_dim=768）
                if len(segment_feat) > self.feature_dim:
                    segment_feat = segment_feat[:self.feature_dim]
                else:
                    segment_feat = np.pad(
                        segment_feat,
                        (0, self.feature_dim - len(segment_feat)),
                        mode='constant'
                    )
                segments.append(segment_feat)

            # 9. 最终检查并返回
            audio_features = np.array(segments)
            audio_features[np.isinf(audio_features)] = 0.0
            audio_features[np.isnan(audio_features)] = 0.0
            return audio_features

        except Exception as e:
            print(f"处理音频特征失败: {e}")
            return np.zeros(CONFIG["audio_feature_dim"])

# 增强的视频处理器
class EnhancedVideoProcessor(VideoProcessor):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        # 添加通道权重机制
        self.feature_weights = np.array([
            1.2, 1.3,  # AU特征权重
            1.5, 1.0, 0.8,  # 头部特征权重
            1.1, 1.4  # 眼部特征权重
        ])  # 提升
    def __init__(self, config):
        super().__init__(config)
        self.config = config
    def _process_au_features(self, participant_id):

        """处理面部动作单元特征"""
        # 在 EnhancedVideoProcessor 中的路径构建
        au_path = os.path.join(self.root_dir, f"{participant_id}_P", f"{participant_id}_CLNF_aus.txt")

        if not os.path.exists(au_path):
            return np.zeros(CONFIG["au_feature_dim"])

        try:
            df = pd.read_csv(au_path)
            # 提取关键AU：4,6,7,9,10,12,14,15,17,23,24
            au_columns = [f"AU{au:02d}_r" for au in [4, 6, 7, 9, 10, 12, 14, 15, 17, 23, 24]]

            # 计算统计特征
            au_mean = df[au_columns].mean().values
            au_freq = (df[au_columns] > 0.5).mean().values

            return np.concatenate([au_mean, au_freq])
        except:
            return np.zeros(CONFIG["au_feature_dim"])

    def _process_eye_features(self, participant_id):
        """处理眼部行为特征"""
        gaze_path = os.path.join(self.root_dir, participant_id,  f"{participant_id}_CLNF_gaze.txt" )


        if not os.path.exists(gaze_path):
            return np.zeros(CONFIG["eye_feature_dim"])

        try:
            df = pd.read_csv(gaze_path)
            features = []

            # 1. 眨眼频率 (次/分钟)
            blink_rate = (df['blink'] == 1).sum() / (len(df) / 60)

            # 2. 平均注视时长
            gaze_duration = df['gaze_duration'].mean()

            # 3. 瞳孔直径变化
            pupil_var = df[['pupil_diam_left', 'pupil_diam_right']].var().mean()

            return np.array([blink_rate, gaze_duration, pupil_var])
        except:
            return np.zeros(CONFIG["eye_feature_dim"])

    def _process_head_pose(self, participant_id):
        """处理头部姿势特征"""
        # 复用CLNF文件的前6列（头部姿态参数）
        clnf_path = os.path.join(self.root_dir, f"{participant_id}_P",
                                 f"{participant_id}_CLNF_features3D.txt")

        if not os.path.exists(clnf_path):
            return np.zeros(CONFIG["pose_feature_dim"])

        try:
            df = self._read_feature_file(clnf_path)
            pose_cols = ['pose_Rx', 'pose_Ry', 'pose_Rz', 'pose_Tx', 'pose_Ty', 'pose_Tz']

            # 计算运动幅度和角度变化
            rotation = df[pose_cols[:3]].std().values
            translation = df[pose_cols[3:]].std().values

            return np.concatenate([rotation, translation])
        except:
            return np.zeros(CONFIG["pose_feature_dim"])

    def process_enhanced_features(self, participant_id):
        base_dir = os.path.join(self.root_dir, f"{participant_id}_P")
        video_txt_path = os.path.join(base_dir, f"{participant_id}_CLNF_features3D.txt")

        if not os.path.exists(video_txt_path):
            print(f"⚠️ 视频文件不存在: {video_txt_path}")
            return np.random.normal(0, 0.1, (16, 8))  # 随机特征作为后备

        try:
            df = read_video_txt(video_txt_path)
            if df.empty:
                return np.random.normal(0, 0.1, (16, 8))

            # 灵活提取特征（不依赖硬编码列名）
            col_names = [col.lower() for col in df.columns]
            y_cols = [i for i, col in enumerate(col_names) if 'y' in col and col.replace('y', '').isdigit()]
            x_cols = [i for i, col in enumerate(col_names) if 'x' in col and col.replace('x', '').isdigit()]
            z_cols = [i for i, col in enumerate(col_names) if 'z' in col and col.replace('z', '').isdigit()]

            # 提取8维核心特征（容错处理）
            core_features = []
            # 1-2: 面部动作单元（简化版）
            if len(y_cols) >= 5:
                core_features.extend([df.iloc[:, y_cols[:5]].mean().mean(), df.iloc[:, y_cols[5:10]].var().mean()])
            else:
                core_features.extend([0.0, 0.0])
            # 3-4: 头部姿态
            if len(z_cols) >= 2:
                core_features.extend([df.iloc[:, z_cols[:2]].mean().mean(), df.iloc[:, z_cols[:2]].std().mean()])
            else:
                core_features.extend([0.0, 0.0])
            # 5-8: 眼部特征（简化版）
            core_features.extend([np.random.normal(0, 0.1) for _ in range(4)])  # 占位

            # 扩展为16段视频特征
            video_features = np.tile(core_features, (16, 1))  # (16, 8)
            # 标准化
            mean = np.mean(video_features, axis=0)
            std = np.std(video_features, axis=0) + 1e-8
            return (video_features - mean) / std

        except Exception as e:
            print(f"处理 {participant_id} 失败: {e}")
            return np.random.normal(0, 0.1, (16, 8))  # 出错时返回随机特征
class DAICWOZDataset(Dataset):
    def __init__(self, df, root_dir):
        self.df = df
        self.root_dir = root_dir
        self.dsm_processor = DeepSeekDSM()
        self.video_processor = EnhancedVideoProcessor(root_dir)  # 修改这一行
        self.audio_processor = AudioProcessor(root_dir)
        self.data = []  # 新增这行# 创建处理器实例前打印调试信息
        print("创建视频处理器...")

        print("创建音频处理器...")


        # 加载数据
        print("开始加载数据...")
        self._load_all_data()
        print("数据加载完成!")


        # self.root_dir = root_dir
        # self.text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        #
        # # 使用新配置
        # self.text_processor = TextProcessor(max_length=CONFIG["max_length"])
        # self.audio_processor = AudioProcessor(root_dir=root_dir)
        # self.video_processor = EnhancedVideoProcessor(root_dir)
        # self.valid_ids = {}  # 用于缓存有效ID

        # # 预加载数据
        # print("预加载参与者数据...")
        # self.data = self._preload_data()

    def _load_all_data(self):
        print(f"预加载{len(self.df)}个参与者的数据...")
        class_counts = [0, 0]  # [非抑郁, 抑郁]
        invalid_samples = 0
        # 新增：统计全零特征样本
        zero_text_count = 0
        zero_video_count = 0
        zero_audio_count = 0

        progress = tqdm(total=len(self.df))

        for idx, row in self.df.iterrows():
            participant_id = str(int(row[CONFIG["id_column"]]))
            # 获取特征
            text_embedding = self.dsm_processor.get_deepseek_dsm(participant_id)
            video_features = self.video_processor.process_enhanced_features(participant_id)
            audio_features = self.audio_processor.process_audio_features(participant_id)

            # 检查有效维度
            if not isinstance(video_features, np.ndarray) or video_features.shape != (CONFIG["video_segments"], 8):
                invalid_samples += 1
                print(f"参与者 {participant_id} 视频维度不合法 {video_features.shape}")
                continue

            # 后续处理（原有逻辑）
            phq8_score = float(row[CONFIG["phq8_column"]])
            label = 1 if phq8_score >= CONFIG["phq8_threshold"] else 0
            class_counts[label] += 1

            self.data.append({
                "participant_id": participant_id,
                "text_emb": text_embedding,
                "video": video_features,
                "audio": audio_features,
                "phq8_score": phq8_score,
                "label": label
            })

            progress.update(1)

        # 新增：打印全零特征统计结果
        print(f"\n【全零特征统计】")
        print(f"  - 文本特征全零样本: {zero_text_count}/{len(self.df)} ({zero_text_count / len(self.df):.2%})")
        print(f"  - 视频特征全零样本: {zero_video_count}/{len(self.df)} ({zero_video_count / len(self.df):.2%})")
        print(f"  - 音频特征全零样本: {zero_audio_count}/{len(self.df)} ({zero_audio_count / len(self.df):.2%})")
        print(f"  - 有效样本: {len(self.data)}, 无效样本: {invalid_samples}")
        print(f"  - 类别分布: 抑郁({class_counts[1]}), 非抑郁({class_counts[0]})")
        progress.close()
    def _validate_files(self, participant_id):
        """验证关键文件是否存在"""
        paths = [
            os.path.join(self.root_dir, participant_id, f"{participant_id}_CLNF_gaze.txt"),
            os.path.join(self.root_dir, participant_id, f"{participant_id}_CLNF_aus.txt"),
            os.path.join(self.root_dir, participant_id, f"{participant_id}_CLNF_pose.txt"),
        ]

        for p in paths:
            if not os.path.exists(p):
                print(f"⚠️ 警告: 文件不存在 {p}")

    def __len__(self):
        return len(self.data) if self.data else 0

    # 在DAICWOZDataset类的__getitem__方法中
    def __getitem__(self, idx):
        # item = self.data[idx]
        # video_features = item['video']
        #
        # # 添加特征有效性验证
        # if np.allclose(video_features, 0, atol=1e-3):
        #     # 生成随机扰动替代特征
        #     video_features = np.random.normal(0, 0.1, (16, 8))
        #     print(f"[特征校正] {item['participant_id']} 全零视频特征被人工替换")
        #
        # return {
        #     # 其他字段保持原样
        #     'video': torch.FloatTensor(video_features)
        # }
        # """获取并处理单个样本，添加严格的无效值检查和处理"""
        # # 1. 获取原始样本（这部分是你原有的代码逻辑）
        # participant_id = self.participant_ids[idx]
        # label = self.labels[idx]
        #
        # # 加载三种模态的特征（假设这些方法存在）
        # text_emb = self._load_text_features(participant_id)
        # video = self._load_video_features(participant_id)
        # audio = self._load_audio_features(participant_id)
        #
        # # 2. 检查并替换特征中的无效值（原始代码逻辑）
        # text_emb = torch.nan_to_num(text_emb, nan=0.0, posinf=1e4, neginf=-1e4)
        # video = torch.nan_to_num(video, nan=0.0, posinf=1e4, neginf=-1e4)
        # audio = torch.nan_to_num(audio, nan=0.0, posinf=1e4, neginf=-1e4)
        #
        # # 3. 记录并处理仍存在的无效值（增强检查）
        # has_invalid = False
        # invalid_log = []
        #
        # if torch.isnan(text_emb).any():
        #     has_invalid = True
        #     invalid_log.append(f"文本特征存在NaN")
        #     text_emb = torch.zeros_like(text_emb)  # 用零张量替换
        #
        # if torch.isnan(video).any():
        #     has_invalid = True
        #     invalid_log.append(f"视频特征存在NaN")
        #     video = torch.zeros_like(video)
        #
        # if torch.isnan(audio).any():
        #     has_invalid = True
        #     invalid_log.append(f"音频特征存在NaN")
        #     audio = torch.zeros_like(audio)
        #
        # # 4. 检查全零特征（新增逻辑）
        # if torch.sum(torch.abs(text_emb)) < 1e-6:
        #     invalid_log.append(f"文本特征几乎全零")
        # if torch.sum(torch.abs(video)) < 1e-6:
        #     invalid_log.append(f"视频特征几乎全零")
        # if torch.sum(torch.abs(audio)) < 1e-6:
        #     invalid_log.append(f"音频特征几乎全零")
        #
        # # 5. 打印详细日志（如果有问题）
        # if has_invalid or invalid_log:
        #     print(f"⚠️ 样本 {participant_id} (标签:{label}) 存在问题: {', '.join(invalid_log)}")
        #
        # # 6. 返回处理后的样本（根据你的原始代码结构调整）
        # return {
        #     'participant_id': participant_id,
        #     'text': text_emb,
        #     'video': video,
        #     'audio': audio,
        #     'label': torch.tensor(label, dtype=torch.long)
        # }
        item = self.data[idx]
        video_features = item['video']

        # 特征有效性验证（保留修正逻辑）
        if np.allclose(video_features, 0, atol=1e-3):
            video_features = np.random.normal(0, 0.1, (16, 8))  # 16段×8维特征
            print(f"[特征校正] {item['participant_id']} 全零视频特征被替换")

        return {
            'participant_id': item["participant_id"],
            'text_emb': torch.FloatTensor(item["text_emb"]),  # 文本特征：(768,)
            'video': torch.FloatTensor(video_features),  # 视频特征：(16, 8)
            'audio': torch.FloatTensor(item["audio"]),  # 音频特征：(16, 768)
            'label': torch.tensor(item["label"], dtype=torch.long)  # 整数标签：0或1
        }
        
# 平衡损失类
class BalancedBCELoss(nn.Module):
    def __init__(self, beta=0.9, eps=1e-5, pos_weight=None):
        super().__init__()
        self.beta = beta
        self.eps = eps
        self.register_buffer('history_weights', torch.zeros(10))
        if pos_weight is not None:
            self.register_buffer('pos_weight_weight', torch.tensor(pos_weight))

    def forward(self, inputs, targets):
        with torch.no_grad():
            pos_ratio = torch.mean(targets).clamp(min=self.eps, max=1 - self.eps)
            current_weight = (1 - pos_ratio) / (pos_ratio + self.eps)

            if hasattr(self, 'pos_weight_weight'):
                current_weight = self.pos_weight_weight / pos_ratio

            self.history_weights = torch.cat([
                self.history_weights[1:],
                current_weight.unsqueeze(0)
            ])

        weight = torch.mean(self.history_weights).clamp(min=0.1, max=10)
        return F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=weight,
            reduction='mean'
        )

# 自定义稳定损失函数
class StableBCELoss(nn.Module):
    """数值稳定的二进制交叉熵损失，特别处理极端输入值"""

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出 (未经过sigmoid激活)
            targets: 目标标签 (one-hot编码)
        """
        # 输入范围限制
        inputs = torch.clamp(inputs, min=-50, max=50)

        # 分离计算两部分
        pos_loss = (1 - targets) * inputs - torch.log1p(torch.exp(inputs)) + self.epsilon
        neg_loss = -targets * inputs - torch.log1p(torch.exp(-inputs)) + self.epsilon

        # 组合损失
        loss_elementwise = - (pos_loss + neg_loss)
        return loss_elementwise.mean()

# 完整的训练函数
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)

    # 关键修正：使用CrossEntropyLoss（适配整数标签）
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    best_val_f1 = -1
    no_improve_count = 0
    training_history = {'train_loss': [], 'train_f1': [], 'val_loss': [], 'val_f1': []}

    # 关闭混合精度（CPU不支持）
    use_amp = False

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        train_preds, train_targets = [], []

        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            # 加载数据
            text_emb = batch['text_emb'].to(device)
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)  # 整数标签：(batch_size,)

            optimizer.zero_grad()

            # 前向传播（不使用混合精度）
            outputs, _ = model(text_emb, video, audio)  # outputs: (batch_size, 2)
            loss = loss_fn(outputs, labels)  # 直接使用整数标签

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # 记录指标
            epoch_train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)  # 预测类别
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 计算训练集指标
        train_loss = epoch_train_loss / len(train_loader)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)
        training_history['train_loss'].append(train_loss)
        training_history['train_f1'].append(train_f1)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                text_emb = batch['text_emb'].to(device)
                video = batch['video'].to(device)
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)

                outputs, _ = model(text_emb, video, audio)
                val_loss += loss_fn(outputs, labels).item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        training_history['val_loss'].append(val_loss)
        training_history['val_f1'].append(val_f1)

        # 打印结果
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"训练: 损失={train_loss:.4f}, F1={train_f1:.4f}")
        print(f"验证: 损失={val_loss:.4f}, F1={val_f1:.4f}")

        # 早停逻辑
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_count = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["model_save_dir"], "best_model.pth"))
            print(f"🌟 保存最佳模型 (F1={best_val_f1:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"⏹ 早停触发（{patience}轮无提升）")
                break

        scheduler.step(val_f1)

    print(f"训练完成! 最佳验证F1: {best_val_f1:.4f}")
    return best_val_f1, training_history

# 模型评估函数
def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    loss_fn = nn.CrossEntropyLoss()  # 与训练一致

    with torch.no_grad():
        for batch in data_loader:
            text_emb = batch['text_emb'].to(device)
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)  # 整数标签

            outputs, _ = model(text_emb, video, audio)
            total_loss += loss_fn(outputs, labels).item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, accuracy, f1

# 主函数
def main():
    # 加载训练数据
    train_dfs = []
    for file in CONFIG["train_files"]:
        if os.path.exists(file):
            df = pd.read_csv(file)
            train_dfs.append(df)

    if not train_dfs:
        print("没有找到有效的训练数据文件")
        return

    train_df = pd.concat(train_dfs, ignore_index=True)

    # 加载测试数据
    if os.path.exists(CONFIG["test_file"]):
        test_df = pd.read_csv(CONFIG["test_file"])
    else:
        print("没有找到测试数据文件")
        return

    # 检查数据完整性
    if CONFIG["id_column"] not in train_df.columns or CONFIG["id_column"] not in test_df.columns:
        print(f"数据框中没有找到ID列: {CONFIG['id_column']}")
        return

    if CONFIG["phq8_column"] not in train_df.columns or CONFIG["phq8_column"] not in test_df.columns:
        print(f"数据框中没有找到PHQ-8列: {CONFIG['phq8_column']}")
        return

    # 创建数据集
    # 创建数据集
    print("准备训练数据...")
    train_dataset = DAICWOZDataset(train_df, CONFIG["daic_woz_root"])
    print(f"数据集加载完成! 有效样本: {len(train_dataset)}")

    # 关键修正：使用分层抽样分割数据集（保持类别比例）
    from sklearn.model_selection import train_test_split
    # 获取所有样本的标签
    labels = [item["label"] for item in train_dataset.data]
    # 分层分割（测试集占15%）
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)),
        test_size=0.15,
        stratify=labels,  # 按标签分层
        random_state=42
    )
    # 创建子集
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)

    # 验证分割后的类别比例
    train_labels = [train_dataset.data[i]["label"] for i in train_indices]
    val_labels = [train_dataset.data[i]["label"] for i in val_indices]
    print(f"训练集 - 抑郁样本: {sum(train_labels)}, 非抑郁: {len(train_labels) - sum(train_labels)}")
    print(f"验证集 - 抑郁样本: {sum(val_labels)}, 非抑郁: {len(val_labels) - sum(val_labels)}")

    # 创建数据加载器
    train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=CONFIG["batch_size"])

    # 删除原始数据集释放内存
    # del train_dataset

    # 初始化模型
    # model = EnhancedMILModel()
    model = EnhancedMILModel()
    print(f"模型创建成功! 视觉处理器输入维度: {model.visual_processor[0].in_features}")

    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # 训练模型
    print("开始训练模型...")
    train_model(model, train_loader, val_loader, epochs=CONFIG["epochs"], lr=CONFIG["learning_rate"])

    # 创建测试数据集
    print("准备测试数据...")
    test_dataset = DAICWOZDataset(test_df, CONFIG["daic_woz_root"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])

    # 加载最佳模型
    print("加载最佳模型...")
    best_model_path = os.path.join(CONFIG["model_save_dir"], CONFIG["best_model"])
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"成功加载最佳模型: {best_model_path}")
    else:
        print("警告: 最佳模型不存在，使用当前模型进行评估")

    # 评估测试集
    print("评估测试集...")
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, nn.BCEWithLogitsLoss(),
                                                  torch.device(CONFIG["device"]))

    # 打印评估结果
    print("\n" + "=" * 60)
    print(f"测试结果 - 准确率: {test_acc:.4f}, F1分数: {test_f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    
