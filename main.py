# main.py
import flwr as fl
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Optional, Callable, Dict, List, Tuple
from torch.utils.data import DataLoader

from flwr.common import Context 

from utils_data import (
    load_dataset_from_parquet,
    prepare_dataset_partitions,
    get_dataloader,
    visualize_client_data_distribution,
    NUM_CLASSES as DATA_NUM_CLASSES,
    IMG_SIZE, 
    Image,
    partition_noniid_custom
)
from model import get_model, NUM_CLASSES as MODEL_NUM_CLASSES
from client import client_fn, NUM_CLASSES as CLIENT_NUM_CLASSES 
from strategy import Krum, CoordMedian 
import json
import torchvision.transforms as T

# --- 类别名称映射 ---
CLASS_ID_TO_NAME_MAPPING = {
    0: "限速 (20km/h)",
    1: "限速 (30km/h)",
    2: "限速 (50km/h)",
    3: "限速 (60km/h)",
    4: "限速 (70km/h)",
    5: "限速 (80km/h)",
    6: "80km/h限速解除",
    7: "限速 (100km/h)",
    8: "限速 (120km/h)",
    9: "禁止超车",
    10: "3.5吨以上车辆禁止超车",
    11: "下一交叉口优先权",
    12: "优先道路",
    13: "让行",
    14: "停车",
    15: "禁止所有车辆通行",
    16: "3.5吨以上车辆禁止通行",
    17: "禁止驶入",
    18: "一般危险警告",
    19: "左弯危险",
    20: "右弯危险",
    21: "双向弯道",
    22: "路面不平",
    23: "路滑",
    24: "右侧道路变窄",
    25: "道路施工",
    26: "交通信号灯",
    27: "行人通行",
    28: "儿童通行",
    29: "自行车通行",
    30: "小心冰雪",
    31: "小心动物通行",
    32: "速度和超车限制解除",
    33: "前方右转 (强制)",
    34: "前方左转 (强制)",
    35: "直行 (强制)",
    36: "直行或右转 (强制)",
    37: "直行或左转 (强制)",
    38: "靠右行驶 (强制)",
    39: "靠左行驶 (强制)",
    40: "强制环岛行驶",
    41: "禁止超车解除",
    42: "3.5吨以上车辆禁止超车解除"
}
# 使用映射生成混淆矩阵的显示标签列表
# 确保标签顺序与 NUM_CLASSES (即类别ID 0 到 NUM_CLASSES-1) 一致
CONFUSION_MATRIX_DISPLAY_LABELS = range(DATA_NUM_CLASSES) # 直接使用数字范围作为标签

# --- 配置 ---
print(f"开始执行 main.py 脚本。当前时间: {datetime.now()}") 
# Data paths
BASE_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if os.path.exists("/home/ps/llm/segmentation/fl/data"):
    BASE_DATA_PATH = "/home/ps/llm/segmentation/fl/data"
elif not os.path.exists(BASE_DATA_PATH):
    print(f"Warning: Local data directory {BASE_DATA_PATH} not found. Trying a generic path, may fallback to dummy data.")
    BASE_DATA_PATH = "data" 


TRAIN_VAL_DATA_SOURCE_PATH = os.path.join(BASE_DATA_PATH, 'train-00000-of-00001.parquet')
DEDICATED_TEST_DATA_PATH = os.path.join(BASE_DATA_PATH, 'test-00000-of-00001.parquet')


USE_VIT_BASE_FOR_ALL_SCENARIOS = False
print(f"模型配置: {'ViT-Base' if USE_VIT_BASE_FOR_ALL_SCENARIOS else '轻量级 ViT'}")

assert DATA_NUM_CLASSES == MODEL_NUM_CLASSES == CLIENT_NUM_CLASSES, "NUM_CLASSES 在模块间不匹配!"
NUM_CLASSES = DATA_NUM_CLASSES
print(f"使用的类别数量 NUM_CLASSES = {NUM_CLASSES}") 

print(f"训练/验证数据源路径: {TRAIN_VAL_DATA_SOURCE_PATH}") 
print(f"专用测试数据路径: {DEDICATED_TEST_DATA_PATH}") 


TOTAL_CLIENTS = 10 
CLIENTS_PER_ROUND = 2 
TOTAL_ROUNDS = 60    
print(f"联邦学习设置: 总客户端数={TOTAL_CLIENTS}, 每轮客户端数={CLIENTS_PER_ROUND}, 总轮数={TOTAL_ROUNDS}") 

CLIENT_CONFIG_BASE = {
    "local_epochs": 3,
    "batch_size": 32,
    "learning_rate": 2e-4,
    "use_vit_base_config": USE_VIT_BASE_FOR_ALL_SCENARIOS,
}

EXPERIMENT_SCENARIOS = {
    "1_Baseline_IID_FedAvg_LightViT": {
        "is_iid": True, "aggregation_strategy": "fedavg", "client_defense": {}, "num_clients": 10
    },  
    "2_LabelFlip_30%_Classes_FedAvg_IID": {
        "is_iid": True, "attack_type": "label_flipping", "attacker_fraction": 0.6,
        "label_flip_type": "max_adversarial", "flip_ratio": 1.0, 
        "targeted_classes": list(range(int(0.3 * NUM_CLASSES))),
        "aggregation_strategy": "fedavg", "client_defense": {}, "num_clients": 10
    },
    # "3_LabelFlip_60%_Classes_FedAvg_IID": {
    #     "is_iid": True, "attack_type": "label_flipping", "attacker_fraction": 0.6,
    #     "label_flip_type": "max_adversarial", "flip_ratio": 1.0,
    #     "targeted_classes": list(range(int(0.6 * NUM_CLASSES))),
    #     "aggregation_strategy": "fedavg", "client_defense": {}, "num_clients": 10
    # },
    "4_NonIID_CustomRichPoor_FedAvg": {
        "is_iid": False,
        "custom_noniid_type": "rich-poor",
        "num_rich_clients": 1,
        "num_classes_per_poor_client": 1,
        "rich_client_class_ratio": 0.8,
        "attack_type": "non_iid_data_as_attack",
        "attacker_fraction": 0.0,
        "aggregation_strategy": "fedavg",
        "client_defense": {},
        "num_clients": 10
    },
    "5_NonIID_CustomRichPoor2_FedAvg": {
        "is_iid": False,
        "custom_noniid_type": "rich-poor",
        "num_rich_clients": 2,
        "num_classes_per_poor_client": 1,
        "rich_client_class_ratio": 0.8,
        "attack_type": "non_iid_data_as_attack",
        "attacker_fraction": 0.0,
        "aggregation_strategy": "fedavg",
        "client_defense": {},
        "num_clients": 10
    },
    "6_CombinedAttack_LabelFlip30%_NonIID_CustomRichPoor_FedAvg": {
        "is_iid": False,
        "custom_noniid_type": "rich-poor",
        "num_rich_clients": 4,
        "num_classes_per_poor_client": 1,
        "rich_client_class_ratio": 0.8,
        "attack_type": "combined",
        "attacker_fraction": 0.6, 
        "label_flip_type": "max_adversarial", 
        "flip_ratio": 1.0,
        "targeted_classes": list(range(int(0.3 * NUM_CLASSES))),
        "aggregation_strategy": "fedavg", 
        "client_defense": {},
        "num_clients": 10
    },
    "7_Defend_Classes_Flip": {
        "is_iid": True, "aggregation_strategy": "fedavg", "client_defense": {}, "num_clients": 10
    },
    "8_Defend_NonIID": {
        "is_iid": True, "aggregation_strategy": "fedavg", "client_defense": {}, "num_clients": 10
    },
    "9_Defend_Class_Flip_NonIID": {
        "is_iid": True, "aggregation_strategy": "fedavg", "client_defense": {}, "num_clients": 10
    },
    # "10_Defend_LabelFlip30%_Classes_Krum": {
    #     "is_iid": True,
    #     "attack_type": "label_flipping",
    #     "attacker_fraction": 0.6,
    #     "label_flip_type": "max_adversarial",
    #     "flip_ratio": 1.0,
    #     "targeted_classes": list(range(int(0.3 * NUM_CLASSES))),
    #     "aggregation_strategy": "krum",
    #     "krum_params": {
    #         "num_malicious_clients": 6,  # 对应 attacker_fraction=0.6
    #         "num_clients_to_aggregate": 4  # 选择最安全的4个客户端
    #     },
    #     "client_defense": {
    #         "gradient_clipping": True,
    #         "clip_norm": 1.0,
    #         "noise_scale": 0.01
    #     },
    #     "num_clients": 10
    # },
    # "11_Defend_NonIID_CustomRichPoor_CoordMedian": {
    #     "is_iid": False,
    #     "custom_noniid_type": "rich-poor",
    #     "num_rich_clients": 1,
    #     "num_classes_per_poor_client": 1,
    #     "rich_client_class_ratio": 0.8,
    #     "attack_type": "non_iid_data_as_attack",
    #     "attacker_fraction": 0.0,
    #     "aggregation_strategy": "coordmedian",
    #     "client_defense": {
    #         "gradient_clipping": True,
    #         "clip_norm": 1.0,
    #         "noise_scale": 0.01,
    #         "adaptive_learning_rate": True,
    #         "min_lr": 1e-5,
    #         "max_lr": 1e-3
    #     },
    #     "num_clients": 10
    # },
    # "12_Defend_CombinedAttack_Krum_CoordMedian": {
    #     "is_iid": False,
    #     "custom_noniid_type": "rich-poor",
    #     "num_rich_clients": 4,
    #     "num_classes_per_poor_client": 1,
    #     "rich_client_class_ratio": 0.8,
    #     "attack_type": "combined",
    #     "attacker_fraction": 0.6,
    #     "label_flip_type": "max_adversarial",
    #     "flip_ratio": 1.0,
    #     "targeted_classes": list(range(int(0.3 * NUM_CLASSES))),
    #     "aggregation_strategy": "krum",
    #     "krum_params": {
    #         "num_malicious_clients": 6,
    #         "num_clients_to_aggregate": 4
    #     },
    #     "client_defense": {
    #         "gradient_clipping": True,
    #         "clip_norm": 1.0,
    #         "noise_scale": 0.01,
    #         "adaptive_learning_rate": True,
    #         "min_lr": 1e-5,
    #         "max_lr": 1e-3,
    #         "label_smoothing": True,
    #         "smoothing_factor": 0.1
    #     },
    #     "num_clients": 10
    # }
}
print(f"已定义 {len(EXPERIMENT_SCENARIOS)} 个实验场景。") 

# --- Data Loading ---
print("正在加载专用的全局测试集...") 
dedicated_test_df = load_dataset_from_parquet(DEDICATED_TEST_DATA_PATH)
dedicated_test_loader = None
if not dedicated_test_df.empty:
    dedicated_test_loader = get_dataloader(dedicated_test_df, batch_size=32, shuffle=False)
    print(f"专用全局测试集加载完毕，样本数: {len(dedicated_test_df)}。DataLoader 已创建。") 
else:
    print(f"警告: 专用全局测试集 ({DEDICATED_TEST_DATA_PATH}) 为空或加载失败。服务器端对测试集的评估将无法进行。") 

# 提前准备固定测试样本，如果专用测试集可用
fixed_test_samples_df_for_eval = None
NUM_SAMPLES_FOR_FIXED_EVAL = 4 # 定义我们希望选取和展示的样本数量

if dedicated_test_loader is not None and not dedicated_test_df.empty:
    temp_fixed_samples_list = []
    # 检查 'label' 列是否存在且至少有一个唯一标签
    if 'label' in dedicated_test_df.columns and dedicated_test_df['label'].nunique() > 0:
        print(f"尝试从专用测试集的不同类别中随机选取最多 {NUM_SAMPLES_FOR_FIXED_EVAL} 个固定样本...")
        unique_labels = dedicated_test_df['label'].unique()
        np.random.shuffle(unique_labels) # 原地打乱标签顺序，以随机选择类别

        # 确定要从多少个不同类别中采样
        num_distinct_classes_to_sample = min(len(unique_labels), NUM_SAMPLES_FOR_FIXED_EVAL)
        
        labels_to_sample_from = unique_labels[:num_distinct_classes_to_sample]

        for lbl in labels_to_sample_from:
            # 确保选取的样本 'image' 列存在且不为空值
            class_specific_samples = dedicated_test_df[
                (dedicated_test_df['label'] == lbl) & (dedicated_test_df['image'].notna())
            ]
            if not class_specific_samples.empty:
                # 从当前类别中随机采样一个样本，为保证每次运行的随机性，使用随机的 random_state
                temp_fixed_samples_list.append(class_specific_samples.sample(1, random_state=np.random.randint(1,100000)))
        
        if temp_fixed_samples_list:
            fixed_test_samples_df_for_eval = pd.concat(temp_fixed_samples_list).reset_index(drop=True)
            print(f"已成功从 {len(fixed_test_samples_df_for_eval)} 个不同类别中选取固定样本。")
        else:
            print("未能从不同类别中选取到任何样本。")

    # 如果按类别采样失败，或者 'label' 列不适合用于采样，则执行回退逻辑
    if fixed_test_samples_df_for_eval is None or fixed_test_samples_df_for_eval.empty:
        if not ('label' in dedicated_test_df.columns and dedicated_test_df['label'].nunique() > 0) :
             print(f"警告: 专用测试集缺少 'label' 列或无唯一标签，无法按类别选取。")
        print(f"回退：尝试选取专用测试集中的前 {NUM_SAMPLES_FOR_FIXED_EVAL} 个有效样本作为固定样本。")
        
        # 在回退逻辑中也确保 'image' 列不为空
        fallback_candidates = dedicated_test_df[dedicated_test_df['image'].notna()]
        if not fallback_candidates.empty:
            num_to_take = min(NUM_SAMPLES_FOR_FIXED_EVAL, len(fallback_candidates))
            fixed_test_samples_df_for_eval = fallback_candidates.iloc[:num_to_take]
        
        if fixed_test_samples_df_for_eval is not None and not fixed_test_samples_df_for_eval.empty:
            print(f"已通过回退逻辑选取了 {len(fixed_test_samples_df_for_eval)} 个固定样本。")
        else:
            print("警告: 回退逻辑也未能选取到任何固定样本。")
            fixed_test_samples_df_for_eval = None # 明确设置为 None

    # 对最终选取的样本进行状态报告
    if fixed_test_samples_df_for_eval is not None and not fixed_test_samples_df_for_eval.empty:
        if len(fixed_test_samples_df_for_eval) < NUM_SAMPLES_FOR_FIXED_EVAL:
            print(f"警告: 最终选取的固定样本数量 ({len(fixed_test_samples_df_for_eval)}) 少于期望的 {NUM_SAMPLES_FOR_FIXED_EVAL} 个。")
        # 确保即使样本少于预期，也打印准备好的数量
        print(f"已准备 {len(fixed_test_samples_df_for_eval)} 个固定测试样本用于每轮评估后的可视化。")
    else:
        # 如果在所有尝试后 fixed_test_samples_df_for_eval 仍然是 None 或空的
        print("警告: 从专用测试集中选取的固定样本为空，轮次评估中的固定样本推理将无法进行。")
        fixed_test_samples_df_for_eval = None # 再次确保是 None
else:
    print("警告: 专用测试集不可用或为空，轮次评估中的固定样本推理将无法进行。")

print(f"正在加载训练/验证数据源: {TRAIN_VAL_DATA_SOURCE_PATH}...") 
source_train_val_df = load_dataset_from_parquet(TRAIN_VAL_DATA_SOURCE_PATH)
df_for_client_partitions = pd.DataFrame()
global_val_df = pd.DataFrame()
global_val_loader = None

if not source_train_val_df.empty:
    print("正在将数据源划分为客户端训练数据和全局验证集...") 
    stratify_col = None
    if 'label' in source_train_val_df.columns and source_train_val_df['label'].nunique() > 1:
        label_counts = source_train_val_df['label'].value_counts()
        if (label_counts < 2).any():
            print(f"警告: 数据集中存在样本数少于2的类别 (总样本数 {len(source_train_val_df)})。分层抽样可能失败。将不使用分层。")
        else:
            stratify_col = source_train_val_df['label']
    else:
        print("警告: 'label' 列不存在或只有一个唯一值，无法进行分层抽样。")

    df_for_client_partitions, global_val_df = train_test_split(
        source_train_val_df,
        test_size=0.2, 
        random_state=42, 
        stratify=stratify_col
    )
    print(f"数据源划分完毕: 客户端训练数据源样本数={len(df_for_client_partitions)}, 全局验证集样本数={len(global_val_df)}") 
    if not global_val_df.empty:
        global_val_loader = get_dataloader(global_val_df, batch_size=32, shuffle=False)
        print("全局验证集 DataLoader 已创建。") 
    else:
        print("警告: 全局验证集为空。服务器端验证将无法进行。") 
else:
    print(f"警告: 训练/验证数据源 {TRAIN_VAL_DATA_SOURCE_PATH} 为空或加载失败。后续实验可能无法进行。") 

def get_evaluate_fn(
    val_loader_for_server: Optional[DataLoader], 
    test_loader_for_server: Optional[DataLoader],
    fixed_test_samples_df_captured: Optional[pd.DataFrame],
    results_dir_for_scenario_captured: str,
    scenario_name_captured: str
    ):
    if val_loader_for_server is None and test_loader_for_server is None:
        print("警告: 全局验证 DataLoader 和全局测试 DataLoader 均不可用，无法创建服务器端评估函数。") 
        return None

    def evaluate_on_server(server_round: int, parameters_ndarrays: fl.common.NDArrays, config: dict):
        model = get_model(use_vit_base_config=USE_VIT_BASE_FOR_ALL_SCENARIOS)
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        model.to(device)

        params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        metrics = {}

        if val_loader_for_server:
            val_correct, val_total, val_loss_sum = 0, 0, 0.0
            val_all_labels, val_all_predictions = [], []
            with torch.no_grad():
                for images, labels in tqdm(val_loader_for_server, desc=f"服务器评估 (验证集) 第 {server_round} 轮", leave=False): 
                    images, labels = images.to(device), labels.to(device).long()
                    outputs = model(images)
                    val_loss_sum += criterion(outputs, labels).item() * images.size(0) 
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_all_labels.extend(labels.cpu().numpy())
                    val_all_predictions.extend(predicted.cpu().numpy())

            val_avg_loss = val_loss_sum / val_total if val_total > 0 else 0.0
            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            val_cm = confusion_matrix(val_all_labels, val_all_predictions, labels=range(NUM_CLASSES))
            metrics.update({
                "val_loss": val_avg_loss,
                "val_accuracy": val_accuracy,
                "val_confusion_matrix": val_cm.tolist() if server_round == TOTAL_ROUNDS else None  # 只在最后一轮保存混淆矩阵
            })
        else:
            # 如果验证加载器不可用，但测试加载器可能可用，因此我们不应在此处为验证指标设置默认值，
            # 除非测试加载器也不可用 (这在函数开始时已检查)。
            # 如果两者都不可用，则此函数不会被调用或返回 None。
            # 如果只有验证加载器不可用，我们将依赖测试指标。
            pass

        if test_loader_for_server:
            test_correct, test_total, test_loss_sum = 0, 0, 0.0
            test_all_labels, test_all_predictions = [], []
            with torch.no_grad():
                for images, labels in tqdm(test_loader_for_server, desc=f"服务器评估 (测试集) 第 {server_round} 轮", leave=False): 
                    images, labels = images.to(device), labels.to(device).long()
                    outputs = model(images)
                    test_loss_sum += criterion(outputs, labels).item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
                    test_all_labels.extend(labels.cpu().numpy())
                    test_all_predictions.extend(predicted.cpu().numpy())

            test_avg_loss = test_loss_sum / test_total if test_total > 0 else 0.0
            test_accuracy = test_correct / test_total if test_total > 0 else 0.0
            test_cm = confusion_matrix(test_all_labels, test_all_predictions, labels=range(NUM_CLASSES))
            metrics.update({
                "test_loss": test_avg_loss,
                "test_accuracy": test_accuracy,
                "test_confusion_matrix": test_cm.tolist() if server_round == TOTAL_ROUNDS else None  # 只在最后一轮保存混淆矩阵
            })
        else:
            # 如果测试加载器不可用，且验证加载器也不可用（如上所述），则函数不会被调用。
            # 如果只有测试加载器不可用，而验证加载器可用，则我们依赖验证指标。
            # 如果两者都不可用，则不应发生这种情况，因为外部检查会阻止。
            pass
        
        # 确保至少有一组指标，以避免键错误
        if not metrics:
             metrics.update({"val_loss": 0.0, "val_accuracy": 0.0, "test_loss": 0.0, "test_accuracy": 0.0})
        elif "val_loss" not in metrics and "test_loss" in metrics: # 只有测试集时
            metrics.update({"val_loss": metrics.get("test_loss", 0.0), "val_accuracy": metrics.get("test_accuracy", 0.0)})
        elif "test_loss" not in metrics and "val_loss" in metrics: # 只有验证集时
             metrics.update({"test_loss": metrics.get("val_loss", 0.0), "test_accuracy": metrics.get("val_accuracy", 0.0)})

        # --- 固定样本推理和可视化 --- 
        if fixed_test_samples_df_captured is not None and not fixed_test_samples_df_captured.empty:
            # 使用已经加载了当前轮次参数的 `model` 和 `device`
            # model.eval() 已经在前面调用过了

            images_for_vis_pil = fixed_test_samples_df_captured['image'].values
            labels_for_vis = fixed_test_samples_df_captured['label'].values

            preprocess_for_inference_vis = T.Compose([
                T.Resize((IMG_SIZE, IMG_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            inputs_for_model_vis_list = []
            for pil_image_for_model in images_for_vis_pil:
                input_tensor = preprocess_for_inference_vis(pil_image_for_model)
                inputs_for_model_vis_list.append(input_tensor)
            
            inputs_for_model_vis = None
            if inputs_for_model_vis_list:
                inputs_for_model_vis = torch.stack(inputs_for_model_vis_list).to(device)
            
            preds_vis = None
            probs_vis = None

            if inputs_for_model_vis is not None:
                with torch.no_grad():
                    outputs_vis = model(inputs_for_model_vis)
                    probs_vis = torch.softmax(outputs_vis, dim=1).cpu().numpy()
                    preds_vis = torch.argmax(outputs_vis, dim=1).cpu().numpy()
            
            num_samples_to_show = min(4, len(fixed_test_samples_df_captured))
            if num_samples_to_show > 0 and preds_vis is not None and probs_vis is not None:
                fig, axes = plt.subplots(1, num_samples_to_show, figsize=(5 * num_samples_to_show, 5), squeeze=False)
                for i, ax_el in enumerate(axes[0]): # 重命名循环变量以避免与外部作用域冲突
                    pil_img_to_display = images_for_vis_pil[i]
                    ax_el.imshow(pil_img_to_display)
                    ax_el.axis('off')
                    ax_el.set_title(
                        f"True: {CLASS_ID_TO_NAME_MAPPING.get(labels_for_vis[i], labels_for_vis[i])}\nPred: {CLASS_ID_TO_NAME_MAPPING.get(preds_vis[i], preds_vis[i])}\nProb: {probs_vis[i][preds_vis[i]]:.2f}",
                        fontsize=10 # 减小字体以适应可能更长的中文标签
                    )
                fig.suptitle(f"{scenario_name_captured} - Round {server_round} - Fixed Test Inference", fontsize=16)
                # results_dir_for_scenario_captured is the main results dir, e.g., fl_results_...
                # scenario_name_captured is the name of the current scenario
                # We need to build the path like: main_results_dir/scenario_name/plot/image.png
                plot_dir_for_fixed_samples = os.path.join(results_dir_for_scenario_captured, scenario_name_captured, "plot")
                # Ensure this plot directory exists (it should be created in the main loop, but a check here is defensive)
                os.makedirs(plot_dir_for_fixed_samples, exist_ok=True)
                vis_path = os.path.join(plot_dir_for_fixed_samples, f"{scenario_name_captured}_round_{server_round}_fixed_test_inference.png")
                plt.savefig(vis_path, bbox_inches='tight')
                plt.close(fig)
                print(f"场景 {scenario_name_captured} 第 {server_round} 轮的固定测试图片推理可视化已保存到: {vis_path}")
            elif num_samples_to_show > 0:
                print(f"场景 {scenario_name_captured} 第 {server_round} 轮: 固定样本推理结果不可用，无法进行可视化。")

        primary_loss = metrics.get("val_loss", metrics.get("test_loss", 0.0)) 
        return primary_loss, metrics
    return evaluate_on_server


def get_strategy(strategy_name, krum_params=None, initial_parameters=None, server_eval_fn=None): 
    if strategy_name == "fedavg":
        return fl.server.strategy.FedAvg(
            fraction_fit=CLIENTS_PER_ROUND / TOTAL_CLIENTS if TOTAL_CLIENTS > 0 else 1.0,
            fraction_evaluate=0.0, 
            min_fit_clients=CLIENTS_PER_ROUND,
            min_evaluate_clients=0, 
            min_available_clients=TOTAL_CLIENTS if TOTAL_CLIENTS > 0 else CLIENTS_PER_ROUND,
            initial_parameters=initial_parameters,
            evaluate_fn=server_eval_fn 
        )
    elif strategy_name == "fedadam":
        return fl.server.strategy.FedAdam(
            fraction_fit=CLIENTS_PER_ROUND / TOTAL_CLIENTS if TOTAL_CLIENTS > 0 else 1.0,
            fraction_evaluate=0.0,
            min_fit_clients=CLIENTS_PER_ROUND,
            min_evaluate_clients=0,
            min_available_clients=TOTAL_CLIENTS if TOTAL_CLIENTS > 0 else CLIENTS_PER_ROUND,
            initial_parameters=initial_parameters,
            evaluate_fn=server_eval_fn,
            beta_1=0.9,
            beta_2=0.999,
            eta=1e-4,
            tau=1e-3
        )
    elif strategy_name == "krum":
        if krum_params is None:
            raise ValueError("Krum 策略必须提供 Krum 参数。") 
        return Krum( 
            fraction_fit=CLIENTS_PER_ROUND / TOTAL_CLIENTS if TOTAL_CLIENTS > 0 else 1.0,
            fraction_evaluate=0.0, 
            min_fit_clients=CLIENTS_PER_ROUND,
            min_evaluate_clients=0,
            min_available_clients=TOTAL_CLIENTS if TOTAL_CLIENTS > 0 else CLIENTS_PER_ROUND,
            num_malicious_clients=krum_params["num_malicious_clients"],
            num_clients_to_aggregate=krum_params["num_clients_to_aggregate"],
            initial_parameters=initial_parameters,
            evaluate_fn=server_eval_fn
        )
    elif strategy_name == "coordmedian":
        return CoordMedian(
            fraction_fit=CLIENTS_PER_ROUND / TOTAL_CLIENTS if TOTAL_CLIENTS > 0 else 1.0,
            fraction_evaluate=0.0, 
            min_fit_clients=CLIENTS_PER_ROUND,
            min_evaluate_clients=0,
            min_available_clients=TOTAL_CLIENTS if TOTAL_CLIENTS > 0 else CLIENTS_PER_ROUND,
            initial_parameters=initial_parameters,
            evaluate_fn=server_eval_fn
        )
    else:
        raise ValueError(f"未知的策略: {strategy_name}") 


# --- Result Analysis & Visualization: Directory Setup ---
# 将结果目录的创建移到主循环之前，确保在循环内可用
results_dir = f"fl_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)
print(f"结果将保存到目录: {results_dir}")

# --- Main Experiment Loop ---
all_experiment_results = {}
scenario_data_prep_results = {} 

if df_for_client_partitions.empty and global_val_df.empty and (dedicated_test_df is None or dedicated_test_df.empty):
    print("错误：所有必要的数据集（客户端训练、全局验证、专用测试）均未能加载或准备。无法继续进行实验。请检查文件路径和数据。") 
    exit()
elif df_for_client_partitions.empty:
    print("错误：用于客户端分区的数据为空。无法继续进行实验。") 
    exit()

def client_fn_wrapper_factory(
    original_client_fn: Callable, 
    all_train_dfs_captured: List[pd.DataFrame],
    val_df_captured: pd.DataFrame,
    client_config_base_captured: Dict,
    attacker_cids_captured: List[int],
    attack_params_captured: Dict,
    defense_params_captured: Dict
) -> Callable[[Context], fl.client.Client]: 
    
    def client_fn_for_simulation(context: Context) -> fl.client.Client: 
        cid_str = ""
        if hasattr(context, 'cid') and context.cid is not None: # Prefer context.cid if available (string)
            cid_str = str(context.cid)
        elif hasattr(context, 'node_id') and context.node_id is not None: # node_id is typically int
            cid_str = str(context.node_id)
        elif hasattr(context, 'partition_id') and context.partition_id is not None: # partition_id is typically int
            cid_str = str(context.partition_id)
        elif context.node_config and "partition-id" in context.node_config: # Fallback to node_config
            cid_str = str(context.node_config["partition-id"])
        elif context.node_config and "cid" in context.node_config:
            cid_str = str(context.node_config["cid"])
        else:
            raise ValueError("Could not determine client ID (cid or partition-id) from context.")

        print("client_id:", cid_str, type(cid_str), "all_train_dfs len:", len(all_train_dfs_captured))

        # 新增：将 client_id 映射到 0~N-1 区间，防止越界
        cid_int = int(cid_str)
        cid_idx = cid_int % len(all_train_dfs_captured)
        print("client_id(raw):", cid_str, "-> mapped idx:", cid_idx, "all_train_dfs len:", len(all_train_dfs_captured))

        return original_client_fn(
            str(cid_idx), 
            all_train_dfs_captured,
            val_df_captured,
            client_config_base_captured,
            attacker_cids_captured,
            attack_params_captured,
            defense_params_captured
        ) 
    return client_fn_for_simulation


for scenario_name, scenario_params in EXPERIMENT_SCENARIOS.items():
    print(f"\n--- 正在运行场景: {scenario_name} ---") 

    # 为当前场景创建特定的结果子目录
    scenario_specific_results_dir = os.path.join(results_dir, scenario_name)
    os.makedirs(scenario_specific_results_dir, exist_ok=True)
    scenario_specific_plot_dir = os.path.join(scenario_specific_results_dir, "plot")
    os.makedirs(scenario_specific_plot_dir, exist_ok=True)
    print(f"场景 {scenario_name} 的结果将保存到: {scenario_specific_results_dir}")
    print(f"场景 {scenario_name} 的绘图将保存到: {scenario_specific_plot_dir}")

    print("正在为客户端准备数据分区...") 
    scenario_num_clients = scenario_params.get("num_clients", TOTAL_CLIENTS)
    if scenario_params.get("custom_noniid_type", None) == "rich-poor":
        client_train_dfs, client_class_counts_scenario = partition_noniid_custom(
            df_for_client_partitions,
            num_clients=scenario_num_clients,
            num_rich_clients=scenario_params.get("num_rich_clients", 1),
            num_classes_per_poor_client=scenario_params.get("num_classes_per_poor_client", 1),
            rich_client_class_ratio=scenario_params.get("rich_client_class_ratio", 0.8)
        )
    else:
        client_train_dfs, client_class_counts_scenario = prepare_dataset_partitions(
            df_for_client_partitions, 
            num_clients=scenario_num_clients,
            is_iid=scenario_params.get("is_iid", True),
            dirichlet_alpha=scenario_params.get("dirichlet_alpha", 0.5)
        )
    scenario_data_prep_results[scenario_name] = {
        "client_train_dfs": client_train_dfs, 
        "client_class_counts": client_class_counts_scenario
    }
    if not client_train_dfs or not any(df is not None and not df.empty for df in client_train_dfs): 
        print(f"错误: 未能为场景 {scenario_name} 准备任何客户端训练数据。跳过此场景。") 
        all_experiment_results[scenario_name] = None 
        continue
    active_clients_with_data = sum(1 for df in client_train_dfs if df is not None and not df.empty)
    print(f"为场景 {scenario_name} 准备了 {active_clients_with_data} 个拥有数据的客户端。") 

    # 在实验开始前可视化客户端数据分布
    print(f"正在为场景 {scenario_name} 生成客户端数据分布图...")
    client_dist_plot_dir = os.path.join(scenario_specific_results_dir, "plot")
    os.makedirs(client_dist_plot_dir, exist_ok=True)
    client_dist_save_path = os.path.join(client_dist_plot_dir, f"{scenario_name}_client_distribution.png")
    visualize_client_data_distribution(
        client_class_counts_scenario,
        num_clients_to_show=TOTAL_CLIENTS,  # 显示所有客户端
        title=f"客户端数据分布: {scenario_name}",
        save_path=client_dist_save_path
    )
    print(f"客户端数据分布图已保存到: {client_dist_save_path}")

    attacker_cids = []
    if scenario_params.get("attack_type") in ["label_flipping", "combined"]:
        num_attackers = int(TOTAL_CLIENTS * scenario_params.get("attacker_fraction", 0.0))
        if num_attackers > 0:
            available_client_ids_for_attack = [i for i, df in enumerate(client_train_dfs) if df is not None and not df.empty]
            if num_attackers > len(available_client_ids_for_attack):
                attacker_cids = available_client_ids_for_attack
            else:
                attacker_cids = np.random.choice(available_client_ids_for_attack, size=num_attackers, replace=False).tolist()

    attack_config_for_client = {}
    if scenario_params.get("attack_type") == "label_flipping" or scenario_params.get("attack_type") == "combined":
        attack_config_for_client["label_flip_type"] = scenario_params.get("label_flip_type", "random_others")
        attack_config_for_client["flip_ratio"] = scenario_params.get("flip_ratio", 1.0)
        attack_config_for_client["targeted_classes"] = scenario_params.get("targeted_classes", None)
        attack_config_for_client["targeted_samples_ratio"] = scenario_params.get("targeted_samples_ratio", 1.0)
        attack_config_for_client["dynamic_flip_schedule"] = scenario_params.get("dynamic_flip_schedule", None)
        attack_config_for_client["preserve_class_distribution"] = scenario_params.get("preserve_class_distribution", False)

    initial_model = get_model(use_vit_base_config=USE_VIT_BASE_FOR_ALL_SCENARIOS)
    initial_parameters_fl = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    )

    server_eval_fn_for_scenario = get_evaluate_fn(
        global_val_loader, 
        dedicated_test_loader,
        fixed_test_samples_df_for_eval, # 传入准备好的固定样本
        results_dir,                    # 传入结果目录
        scenario_name                   # 传入场景名称
        )

    strategy = get_strategy(
        scenario_params["aggregation_strategy"],
        krum_params=scenario_params.get("krum_params"),
        initial_parameters=initial_parameters_fl,
        server_eval_fn=server_eval_fn_for_scenario
    )
    
    current_simulation_client_fn = client_fn_wrapper_factory(
        original_client_fn=client_fn,
        all_train_dfs_captured=client_train_dfs,
        val_df_captured=pd.DataFrame(),
        client_config_base_captured=CLIENT_CONFIG_BASE, # Pass CLIENT_CONFIG_BASE
        attacker_cids_captured=attacker_cids,
        attack_params_captured=attack_config_for_client,
        defense_params_captured=scenario_params.get("client_defense", {})
    )

    print(f"开始场景 {scenario_name} 的 Flower 模拟，共 {TOTAL_ROUNDS} 轮...") 
    history = None
    try:
        num_sim_clients = TOTAL_CLIENTS
        history = fl.simulation.start_simulation(
            client_fn=current_simulation_client_fn, 
            num_clients=num_sim_clients, 
            config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
            strategy=strategy,
            # 强制客户端在 CPU 上运行以进行调试
            client_resources={"num_cpus": 1, "num_gpus": 1} # 恢复并设置GPU资源
        )
        all_experiment_results[scenario_name] = history
        print(f"--- 场景 {scenario_name} 模拟完成 ---") 

        final_metrics_summary = {}
        if history and history.metrics_centralized: 
            for metric_name, values in history.metrics_centralized.items():
                if values: 
                    final_metrics_summary[metric_name] = values[-1][1] 
        if history and history.losses_centralized: 
             if history.losses_centralized :
                final_metrics_summary["primary_loss_server"] = history.losses_centralized[-1][1]

        if final_metrics_summary:
            print(f"场景 {scenario_name} 服务器端最终评估摘要: {final_metrics_summary}") 
        else:
            print(f"场景 {scenario_name}: 未找到服务器端集中评估的最终度量。") 

        # 保存每轮的指标到 CSV/JSON
        if history is not None and history.metrics_centralized:
            metrics_per_round = []
            # 假设所有指标的轮次对齐
            rounds = [item[0] for item in history.metrics_centralized.get("val_accuracy", [])]
            for i, rnd in enumerate(rounds):
                record = {"round": rnd}
                for metric_name, values in history.metrics_centralized.items():
                    if len(values) > i:
                        record[metric_name] = values[i][1]
                metrics_per_round.append(record)
            # 保存为 CSV
            df_metrics = pd.DataFrame(metrics_per_round)
            metrics_csv_path = os.path.join(scenario_specific_results_dir, f"{scenario_name}_metrics.csv")
            df_metrics.to_csv(metrics_csv_path, index=False, encoding='utf-8-sig')
            print(f"场景 {scenario_name} 的每轮指标已保存到: {metrics_csv_path}")
            # 如需保存混淆矩阵等复杂结构，也可保存为 JSON
            metrics_json_path = os.path.join(scenario_specific_results_dir, f"{scenario_name}_metrics.json")
            with open(metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(metrics_per_round, f, ensure_ascii=False, indent=2)
            print(f"场景 {scenario_name} 的每轮指标（含混淆矩阵）已保存到: {metrics_json_path}")

    except Exception as e:
        print(f"场景 {scenario_name} 模拟过程中发生严重错误: {e}") 
        all_experiment_results[scenario_name] = None 
        import traceback
        traceback.print_exc()

# --- Result Analysis & Visualization ---
print("\n\n--- 所有实验已完成。开始分析结果 ---") 

summary_table_data = []

# ACCURACY PLOT
fig_acc, ax_acc_val = plt.subplots(figsize=(15, 8), dpi=120) # Adjusted figsize
ax_acc_test = ax_acc_val.twinx() 
lines_val_acc, labels_val_acc = [], []
lines_test_acc, labels_test_acc = [], []

for scenario_name, history in all_experiment_results.items():
    if history is None:
        summary_table_data.append({
            "场景": scenario_name, "IID": "N/A", "攻击": "N/A",
            "聚合策略": "N/A", "客户端防御": "N/A",
            "最终验证集准确率": "失败", "最终测试集准确率": "失败"
        })
        continue
    
    val_accuracies = history.metrics_centralized.get("val_accuracy", [])
    test_accuracies = history.metrics_centralized.get("test_accuracy", [])
    final_val_acc_val, final_test_acc_val = 0.0, 0.0

    if val_accuracies:
        rounds = [item[0] for item in val_accuracies]
        accs = [item[1] for item in val_accuracies]
        line, = ax_acc_val.plot(rounds, accs, marker='o', linestyle='-', label=f"{scenario_name} (Val Acc)") 
        lines_val_acc.append(line)
        labels_val_acc.append(f"{scenario_name} (Val Acc)")
        final_val_acc_val = accs[-1] if accs else 0.0
    if test_accuracies:
        rounds = [item[0] for item in test_accuracies]
        accs = [item[1] for item in test_accuracies]
        line, = ax_acc_test.plot(rounds, accs, marker='x', linestyle='--', label=f"{scenario_name} (Test Acc)")
        lines_test_acc.append(line)
        labels_test_acc.append(f"{scenario_name} (Test Acc)")
        final_test_acc_val = accs[-1] if accs else 0.0
    
    current_scenario_params = EXPERIMENT_SCENARIOS[scenario_name]
    attack_desc = f"{current_scenario_params.get('attack_type', '无')}"
    if current_scenario_params.get('attacker_fraction', 0) > 0:
        attack_desc += f" ({current_scenario_params.get('attacker_fraction')*100:.0f}%)"
    summary_table_data.append({
        "场景": scenario_name, "IID": "是" if current_scenario_params.get("is_iid", False) else "否", 
        "攻击": attack_desc, "聚合策略": current_scenario_params.get("aggregation_strategy", "N/A"), 
        "客户端防御": "有" if current_scenario_params.get("client_defense") else "无", 
        "最终验证集准确率": f"{final_val_acc_val:.4f}", "最终测试集准确率": f"{final_test_acc_val:.4f}"  
    })

ax_acc_val.set_xlabel('联邦学习轮次', fontsize=12) 
ax_acc_val.set_ylabel('验证集准确率', color='tab:blue', fontsize=12) 
ax_acc_test.set_ylabel('测试集准确率', color='tab:orange', fontsize=12) 
ax_acc_val.tick_params(axis='y', labelcolor='tab:blue', labelsize=10)
ax_acc_test.tick_params(axis='y', labelcolor='tab:orange', labelsize=10)
ax_acc_val.tick_params(axis='x', labelsize=10)
fig_acc.suptitle('全局模型准确率随轮次变化', fontsize=16) 

combined_lines_acc = lines_val_acc + lines_test_acc
combined_labels_acc = labels_val_acc + labels_test_acc
if combined_lines_acc:
    # Position legend to the right of the plot
    fig_acc.legend(combined_lines_acc, combined_labels_acc, loc='center left', bbox_to_anchor=(0.98, 0.5), fontsize=9, title="Scenarios")
    # Adjust subplot to make room for legend
    fig_acc.subplots_adjust(right=0.75) # May need to adjust this value (e.g., 0.7, 0.65) depending on legend width

ax_acc_val.grid(True, linestyle='--', alpha=0.7)
accuracy_plot_path = os.path.join(results_dir, "global_accuracy_comparison.png")
plt.savefig(accuracy_plot_path, bbox_inches='tight') 
plt.close(fig_acc)


# LOSS PLOT
fig_loss, ax_loss_val = plt.subplots(figsize=(15, 8), dpi=120) # Adjusted figsize
ax_loss_test = ax_loss_val.twinx()
lines_val_loss, labels_val_loss = [], []
lines_test_loss, labels_test_loss = [], []

for scenario_name, history in all_experiment_results.items():
    if history is None: continue
    val_losses_data = history.metrics_centralized.get("val_loss", []) 
    test_losses_data = history.metrics_centralized.get("test_loss", []) 
    if val_losses_data : 
        rounds = [item[0] for item in val_losses_data]
        losses = [item[1] for item in val_losses_data]
        line, = ax_loss_val.plot(rounds, losses, marker='o', linestyle='-', label=f"{scenario_name} (Val Loss)") 
        lines_val_loss.append(line)
        labels_val_loss.append(f"{scenario_name} (Val Loss)")
    if test_losses_data: 
        rounds = [item[0] for item in test_losses_data]
        losses = [item[1] for item in test_losses_data]
        line, = ax_loss_test.plot(rounds, losses, marker='x', linestyle='--', label=f"{scenario_name} (Test Loss)") 
        lines_test_loss.append(line)
        labels_test_loss.append(f"{scenario_name} (Test Loss)")

ax_loss_val.set_xlabel('联邦学习轮次', fontsize=12) 
ax_loss_val.set_ylabel('验证集损失', color='tab:blue', fontsize=12) 
ax_loss_test.set_ylabel('测试集损失', color='tab:orange', fontsize=12) 
ax_loss_val.tick_params(axis='y', labelcolor='tab:blue', labelsize=10)
ax_loss_test.tick_params(axis='y', labelcolor='tab:orange', labelsize=10)
ax_loss_val.tick_params(axis='x', labelsize=10)
fig_loss.suptitle('全局模型损失随轮次变化', fontsize=16) 

combined_lines_loss = lines_val_loss + lines_test_loss
combined_labels_loss = labels_val_loss + labels_test_loss
if combined_lines_loss:
    fig_loss.legend(combined_lines_loss, combined_labels_loss, loc='center left', bbox_to_anchor=(0.98, 0.5), fontsize=9, title="Scenarios")
    fig_loss.subplots_adjust(right=0.75) # Adjust for legend

ax_loss_val.grid(True, linestyle='--', alpha=0.7)
loss_plot_path = os.path.join(results_dir, "global_loss_comparison.png")
plt.savefig(loss_plot_path, bbox_inches='tight') 
plt.close(fig_loss)


# CONFUSION MATRIX PLOTS
for scenario_name, history in all_experiment_results.items():
    if history is None or not history.metrics_centralized:
        continue
    val_cm_data = history.metrics_centralized.get("val_confusion_matrix", [])
    if val_cm_data:
        last_round_val_cm_list = val_cm_data[-1][1] 
        try:
            cm_array_val = np.array(last_round_val_cm_list)
            if cm_array_val.shape == (NUM_CLASSES, NUM_CLASSES):
                fig_cm_val, ax_cm_val = plt.subplots(figsize=(13, 11), dpi=100) # 恢复调整 figsize 和 dpi
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_array_val, display_labels=CONFUSION_MATRIX_DISPLAY_LABELS)
                disp.plot(cmap=plt.cm.Blues, ax=ax_cm_val, xticks_rotation='vertical', values_format='d') 
                ax_cm_val.set_title(f'{scenario_name}\\n验证集混淆矩阵 (最终轮)', fontsize=12) 
                val_cm_plot_dir = os.path.join(results_dir, scenario_name, "plot") # results_dir is top level
                os.makedirs(val_cm_plot_dir, exist_ok=True) # Ensure it exists
                cm_plot_path_val = os.path.join(val_cm_plot_dir, f"{scenario_name}_VAL_confusion_matrix.png")
                plt.savefig(cm_plot_path_val, bbox_inches='tight') 
                plt.close(fig_cm_val)
        except Exception as e:
            print(f"绘制场景 {scenario_name} 的验证集混淆矩阵时出错: {e}") 

    test_cm_data = history.metrics_centralized.get("test_confusion_matrix", []) 
    if test_cm_data:
        last_round_test_cm_list = test_cm_data[-1][1]
        try:
            cm_array_test = np.array(last_round_test_cm_list)
            if cm_array_test.shape == (NUM_CLASSES, NUM_CLASSES):
                fig_cm_test, ax_cm_test = plt.subplots(figsize=(13, 11), dpi=100) # 恢复调整 figsize 和 dpi
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_array_test, display_labels=CONFUSION_MATRIX_DISPLAY_LABELS)
                disp.plot(cmap=plt.cm.Blues, ax=ax_cm_test, xticks_rotation='vertical', values_format='d')
                ax_cm_test.set_title(f'{scenario_name}\\n测试集混淆矩阵 (最终轮)', fontsize=12) 
                test_cm_plot_dir = os.path.join(results_dir, scenario_name, "plot") # results_dir is top level
                os.makedirs(test_cm_plot_dir, exist_ok=True) # Ensure it exists
                cm_plot_path_test = os.path.join(test_cm_plot_dir, f"{scenario_name}_TEST_confusion_matrix.png")
                plt.savefig(cm_plot_path_test, bbox_inches='tight') 
                plt.close(fig_cm_test)
        except Exception as e:
            print(f"绘制场景 {scenario_name} 的测试集混淆矩阵时出错: {e}") 

# CLIENT DATA DISTRIBUTION PLOTS
for scenario_name, scenario_params in EXPERIMENT_SCENARIOS.items():
    if not scenario_params.get("is_iid", True) and scenario_name in scenario_data_prep_results:
        prep_data = scenario_data_prep_results[scenario_name]
        if prep_data and "client_class_counts" in prep_data:
            client_class_counts_viz = prep_data['client_class_counts']
            if client_class_counts_viz: 
                try:
                    # Call visualize_client_data_distribution (ensure it uses bbox_inches='tight' if saving)
                    # visualize_client_data_distribution in utils_data.py also needs adjustment for this
                    client_dist_plot_dir = os.path.join(results_dir, scenario_name, "plot") # results_dir is top level
                    os.makedirs(client_dist_plot_dir, exist_ok=True) # Ensure it exists
                    client_dist_save_path = os.path.join(client_dist_plot_dir, f"{scenario_name}_client_distribution.png")
                    visualize_client_data_distribution(
                        client_class_counts_viz,
                        num_clients_to_show=min(10, TOTAL_CLIENTS),
                        title=f"客户端数据分布: {scenario_name}", 
                        save_path=client_dist_save_path
                    )
                except Exception as e:
                    print(f"为场景 {scenario_name} 可视化客户端分布时出错: {e}") 

summary_df = pd.DataFrame(summary_table_data)
print("\n--- 实验总结 ---") 
try:
    from tabulate import tabulate
    print(tabulate(summary_df, headers='keys', tablefmt='grid'))
except ImportError:
    print(summary_df.to_string())

summary_csv_path = os.path.join(results_dir, "experiment_summary.csv")
summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig') 
print(f"总结表已保存到: {summary_csv_path}") 

print(f"\n所有结果和绘图均保存在目录: {results_dir}") 
print(f"脚本执行结束。当前时间: {datetime.now()}")