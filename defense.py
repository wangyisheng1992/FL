import torch
import copy
import numpy as np
from typing import List, Dict, Tuple

def validate_model_update(
    current_update: List[np.ndarray],
    history_updates: List[List[np.ndarray]],
    threshold: float = 0.1
) -> bool:
    """
    验证模型更新是否异常
    Args:
        current_update: 当前模型更新
        history_updates: 历史模型更新列表
        threshold: 异常阈值
    Returns:
        bool: 更新是否正常
    """
    if not history_updates:
        return True
        
    # 计算当前更新与历史更新的平均差异
    diffs = []
    for hist_update in history_updates:
        diff = np.mean([np.abs(c - h).mean() for c, h in zip(current_update, hist_update)])
        diffs.append(diff)
    
    avg_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    
    # 如果平均差异或最大差异超过阈值，认为是异常
    return avg_diff < threshold and max_diff < threshold * 2

def detect_gradient_anomaly(
    current_gradients: List[np.ndarray],
    history_gradients: List[List[np.ndarray]],
    std_threshold: float = 3.0
) -> bool:
    """
    检测梯度是否异常
    Args:
        current_gradients: 当前梯度
        history_gradients: 历史梯度列表
        std_threshold: 标准差阈值
    Returns:
        bool: 是否检测到异常
    """
    if not history_gradients:
        return False
        
    # 计算当前梯度的统计特征
    current_norm = np.mean([np.linalg.norm(g) for g in current_gradients])
    current_mean = np.mean([np.mean(g) for g in current_gradients])
    current_std = np.mean([np.std(g) for g in current_gradients])
    
    # 计算历史梯度的统计特征
    history_norms = [np.mean([np.linalg.norm(g) for g in gs]) for gs in history_gradients]
    history_means = [np.mean([np.mean(g) for g in gs]) for gs in history_gradients]
    history_stds = [np.mean([np.std(g) for g in gs]) for gs in history_gradients]
    
    # 计算与历史统计量的差异
    norm_diff = abs(current_norm - np.mean(history_norms)) / (np.std(history_norms) + 1e-10)
    mean_diff = abs(current_mean - np.mean(history_means)) / (np.std(history_means) + 1e-10)
    std_diff = abs(current_std - np.mean(history_stds)) / (np.std(history_stds) + 1e-10)
    
    # 如果任一统计量差异超过阈值，认为是异常
    return norm_diff > std_threshold or mean_diff > std_threshold or std_diff > std_threshold

def clip_gradients(
    gradients: List[torch.Tensor],
    max_norm: float = 1.0
) -> List[torch.Tensor]:
    """
    对梯度进行裁剪
    Args:
        gradients: 梯度列表
        max_norm: 最大范数
    Returns:
        List[torch.Tensor]: 裁剪后的梯度
    """
    total_norm = 0.0
    for grad in gradients:
        if grad is not None:
            total_norm += grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in gradients:
            if grad is not None:
                grad.data.mul_(clip_coef)
                
    return gradients

# Krum 聚合算法：选择与其他客户端参数最接近的一个客户端
def aggregate_weights_krum(updates: Dict[str, List[np.ndarray]], num_adversaries: int = 1) -> Dict[str, List[np.ndarray]]:
    """
    使用Krum算法聚合模型更新
    Args:
        updates: 客户端更新字典
        num_adversaries: 恶意客户端数量
    Returns:
        Dict[str, List[np.ndarray]]: 聚合后的更新
    """
    distances = {}
    updates_list = list(updates.values())
    names = list(updates.keys())

    for i in range(len(updates_list)):
        dist = 0
        for j in range(len(updates_list)):
            if i != j:
                # 计算第i个和第j个客户端参数的欧氏距离平方和
                dist += sum([(u - v).norm().item() ** 2 for u, v in zip(updates_list[i], updates_list[j])])
        distances[i] = dist

    selected_idx = min(distances, key=distances.get)
    print(f"Krum聚合选择的客户端: {names[selected_idx]}")
    return {names[selected_idx]: updates_list[selected_idx]}

# Multi-Krum 聚合算法：选择多个最接近的客户端并求平均
def aggregate_weights_multi_krum(
    updates: Dict[str, List[np.ndarray]],
    m: int = 3,
    num_adversaries: int = 1
) -> Dict[str, List[np.ndarray]]:
    """
    使用Multi-Krum算法聚合模型更新
    Args:
        updates: 客户端更新字典
        m: 选择的客户端数量
        num_adversaries: 恶意客户端数量
    Returns:
        Dict[str, List[np.ndarray]]: 聚合后的更新
    """
    updates_list = list(updates.values())
    names = list(updates.keys())
    scores = []

    for i in range(len(updates_list)):
        dists = []
        for j in range(len(updates_list)):
            if i != j:
                # 计算第i个和第j个客户端参数的欧氏距离平方和
                dist = sum([(u - v).norm().item() ** 2 for u, v in zip(updates_list[i], updates_list[j])])
                dists.append(dist)
        dists.sort()
        # 只取与最近的(n-恶意客户端-1)个的距离之和
        scores.append(sum(dists[:len(updates_list) - num_adversaries - 1]))

    # 选择得分最低的m个客户端
    top_indices = sorted(range(len(scores)), key=lambda x: scores[x])[:m]
    selected_updates = [updates_list[i] for i in top_indices]
    print(f"Multi-Krum聚合选择的客户端索引: {top_indices}")
    
    # 计算加权平均
    weights = [1.0 / (scores[i] + 1e-10) for i in top_indices]
    weights = np.array(weights) / np.sum(weights)
    
    avg_update = [
        np.sum([update[layer_idx] * w for update, w in zip(selected_updates, weights)], axis=0)
        for layer_idx in range(len(selected_updates[0]))
    ]
    
    return {"aggregated": avg_update}