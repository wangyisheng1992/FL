# attacks.py
import numpy as np
import random

# 从 utils_data.py 导入 NUM_CLASSES 或在此处定义
NUM_CLASSES = 43

# 定义标签翻转映射 (基于您的实验设计和误分类图谱)
# 示例: {source_class: target_class}
LABEL_FLIP_MAPPINGS = {
    # 容易混淆的类别：
    # 例如：数字相近的限速标志，或形状、颜色相似但细节不同的标志
    "high_confusion": {
        0: 1,   # 限速20km/h 翻转到 限速30km/h (数字相邻，最容易混淆)
        2: 3,   # 限速50km/h 翻转到 限速60km/h (数字相邻)
        9: 10,  # 禁止超车 翻转到 3.5吨以上车辆禁止超车 (仅多一个卡车图标)
        33: 34, # 前方右转 翻转到 前方左转 (蓝色圆形，箭头方向相反)
        19: 20  # 左弯危险 翻转到 右弯危险 (黄色三角形，曲线方向相反)
    },
    # 不容易混淆的类别：
    # 例如：形状、颜色或内部符号差异巨大的标志
    "low_confusion": {
        0: 13,  # 限速20km/h (圆形) 翻转到 让行 (倒三角形，形状差异大)
        5: 14,  # 限速80km/h (圆形) 翻转到 停车 (八边形，形状差异大)
        2: 12,  # 限速50km/h (圆形) 翻转到 优先道路 (黄色菱形，形状颜色差异大)
        9: 18,  # 禁止超车 (圆形带斜线) 翻转到 一般危险警告 (三角形带感叹号，符号差异大)
        35: 25  # 直行 (蓝色圆形箭头) 翻转到 道路施工 (黄色三角形带工人，图案差异大)
    },
    # 中等混淆的类别：
    # 既不是最相似也不是最不相似，可能同属一类但差异稍大，或跨类别但有轻微视觉关联
    "medium_confusion": {
        0: 5,   # 限速20km/h 翻转到 限速80km/h (同是限速，但数字相隔较远)
        4: 7,   # 限速70km/h 翻转到 限速100km/h (同是限速，数字相隔较远)
        15: 17, # 禁止所有车辆通行 翻转到 禁止驶入 (都是圆形禁止标志，内部图案有区别)
        36: 37  # 直行或右转 翻转到 直行或左转 (蓝色圆形，箭头组合方向不同)
    },
    # 最大对抗性翻转：翻转到与原类别最不相似的类别，最大化分类错误
    "max_adversarial": {
        0: 42,  # 限速20km/h 翻转到 环形交叉路口终止（完全不同的概念）
        1: 38,  # 限速30km/h 翻转到 直行靠右行驶（完全不同的语义）
        2: 17,  # 限速50km/h 翻转到 禁止驶入（禁令与限速对立）
        3: 15,  # 限速60km/h 翻转到 禁止所有车辆通行（限速与禁行对立）
        4: 26,  # 限速70km/h 翻转到 注意行人（警告与限制混淆）
        5: 40,  # 限速80km/h 翻转到 环形交叉路口（指示与限制混淆）
        6: 32,  # 解除限速80km/h 翻转到 解除所有超车限制（语义相近但功能不同）
        7: 11,  # 限速100km/h 翻转到 右侧通行（指示与限制混淆）
        8: 16,  # 限速120km/h 翻转到 停车让行（完全相反的指令）
        9: 41,  # 禁止超车 翻转到 公交专用道（完全不同的交通概念）
        10: 36, # 3.5吨以上车辆禁止超车 翻转到 直行或右转（禁令与指示对立）
        # 其他类别可以按照类似逻辑添加...
    },
    # 特定类别集中攻击：只攻击少数关键类别，最大化对特定场景的危害
    "critical_safety": {
        # 只翻转安全关键类别，如警告和禁止类标志
        13: 14,  # 让行 翻转到 停车（错误的优先权判断，可能导致碰撞）
        14: 0,   # 停车 翻转到 限速20km/h（完全忽略停车指令）
        17: 35,  # 禁止驶入 翻转到 直行（引导车辆逆行）
        18: 7,   # 一般危险警告 翻转到 限速100km/h（忽略危险，增加速度）
        26: 40,  # 注意行人 翻转到 环形交叉路口（忽略行人安全）
        27: 5,   # 行人通行 翻转到 限速80km/h（行人区域高速行驶）
        19: 5,   # 左弯危险 翻转到 限速80km/h（在危险弯道处高速行驶）
        20: 7,   # 右弯危险 翻转到 限速100km/h（在危险弯道处高速行驶）
    },
    # 完全随机翻转：完全随机地翻转到任何类别，包括可能的合理和不合理翻转
    "completely_random": "random",
    
    # 定向混淆翻转：所有类别都故意翻转到类似但错误的类别
    "directed_confusion": {
        # 所有限速标志互相混淆
        0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 7: 8, 8: 0,
        # 所有警告标志互相混淆
        18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 18,
        # 所有禁止标志互相混淆
        9: 10, 10: 15, 15: 16, 16: 17, 17: 9,
        # 所有指示标志互相混淆
        33: 34, 34: 35, 35: 36, 36: 37, 37: 38, 38: 39, 39: 40, 40: 41, 41: 33
    }
}

# 添加对所有类别的完整映射，确保所有类别都有翻转目标
def create_full_mapping(mapping_type="random"):
    """创建覆盖所有类别的完整映射"""
    full_map = {}
    
    if mapping_type == "worst_case":
        # 创建到最容易混淆类别的映射
        # 这里简单示例，实际中可以基于域知识或模型混淆矩阵来确定
        for i in range(NUM_CLASSES):
            if i < NUM_CLASSES - 1:
                full_map[i] = i + 1
            else:
                full_map[i] = 0
                
    elif mapping_type == "max_distance":
        # 翻转到"语义距离"最远的类别
        # 这里简单实现为翻转到NUM_CLASSES/2距离的类别
        for i in range(NUM_CLASSES):
            full_map[i] = (i + NUM_CLASSES//2) % NUM_CLASSES
            
    elif mapping_type == "single_target":
        # 所有类别都翻转到同一个目标类别（如类别0）
        target = random.randint(0, NUM_CLASSES-1)
        for i in range(NUM_CLASSES):
            if i != target:  # 避免自身映射
                full_map[i] = target
            else:
                full_map[i] = (target + 1) % NUM_CLASSES
                
    elif mapping_type == "cluster_targets":
        # 将所有类别映射到少数几个目标类别
        num_targets = max(1, NUM_CLASSES // 10)  # 使用约10%的类别作为目标
        targets = random.sample(range(NUM_CLASSES), num_targets)
        for i in range(NUM_CLASSES):
            if i not in targets:
                full_map[i] = random.choice(targets)
            else:
                # 如果当前类别是目标类别之一，则映射到另一个目标
                other_targets = [t for t in targets if t != i]
                if other_targets:
                    full_map[i] = random.choice(other_targets)
                else:
                    # 如果只有一个目标类别，则映射到非目标类别
                    non_targets = [j for j in range(NUM_CLASSES) if j not in targets]
                    full_map[i] = random.choice(non_targets) if non_targets else (i + 1) % NUM_CLASSES
            
    else:  # random
        # 随机映射，但确保每个类别都有唯一的目标
        targets = list(range(NUM_CLASSES))
        random.shuffle(targets)
        for i in range(NUM_CLASSES):
            if targets[i] == i:  # 避免映射到自身
                # 交换与下一个类别
                j = (i + 1) % NUM_CLASSES
                targets[i], targets[j] = targets[j], targets[i]
            full_map[i] = targets[i]
            
    return full_map

class LabelFlipper:
    def __init__(self, flip_type="high_confusion", custom_mapping=None, flip_ratio=1.0, targeted_classes=None, 
                 targeted_samples_ratio=1.0, dynamic_flip_schedule=None, preserve_class_distribution=False):
        """
        初始化标签翻转器
        
        参数:
            flip_type: 翻转类型，可选值包括:
                "high_confusion", "low_confusion", "medium_confusion", 
                "max_adversarial", "critical_safety", "completely_random",
                "directed_confusion", "worst_case", "max_distance", 
                "single_target", "cluster_targets", "full_random"
            custom_mapping: 自定义映射字典
            flip_ratio: 标签翻转比例，范围[0,1]，表示有多少比例的样本会被翻转
            targeted_classes: 目标类别列表，仅这些类别的标签会被考虑翻转
            targeted_samples_ratio: 在目标类别中，有多少比例的样本会被翻转，范围[0,1]
            dynamic_flip_schedule: 动态翻转调度，格式为 {轮次: 翻转率}，例如 {0: 0.1, 5: 0.5, 10: 0.9}
            preserve_class_distribution: 是否保持类别分布不变（翻转后每个类别的样本数与翻转前相同）
        """
        self.flip_ratio = max(0.0, min(1.0, flip_ratio))  # 确保在[0,1]范围内
        self.targeted_classes = set(targeted_classes) if targeted_classes else None
        self.targeted_samples_ratio = max(0.0, min(1.0, targeted_samples_ratio))
        self.dynamic_flip_schedule = dynamic_flip_schedule
        self.current_round = 0
        self.preserve_class_distribution = preserve_class_distribution
        self.original_class_distribution = None  # 将在flip_dataset中设置
        self.class_flip_targets = {}  # 保存每个类别应该翻转到哪些目标类别的映射
        
        # 设置映射
        if custom_mapping:
            self.mapping = custom_mapping
        elif flip_type in LABEL_FLIP_MAPPINGS:
            # 处理"completely_random"特殊情况
            if flip_type == "completely_random" or LABEL_FLIP_MAPPINGS[flip_type] == "random":
                self.mapping = {"type": "random_others"}
            else:
                self.mapping = LABEL_FLIP_MAPPINGS[flip_type]
        elif flip_type == "worst_case":
            self.mapping = create_full_mapping("worst_case")
        elif flip_type == "max_distance":
            self.mapping = create_full_mapping("max_distance")
        elif flip_type == "single_target":
            self.mapping = create_full_mapping("single_target")
        elif flip_type == "cluster_targets":
            self.mapping = create_full_mapping("cluster_targets")
        elif flip_type == "full_random":
            self.mapping = create_full_mapping("random")
        else:
            # 默认随机翻转到非自身的其他类别
            print(f"警告: 翻转类型 '{flip_type}' 未识别。使用随机非自身翻转作为后备方案。")
            self.mapping = {"type": "random_others"}

        self.flip_type = flip_type
        
        if self.mapping.get("type") == "random_others":
            print(f"标签翻转器初始化为随机非自身翻转，翻转比例: {self.flip_ratio}")
        else:
            print(f"标签翻转器初始化，类型: {flip_type}，翻转比例: {self.flip_ratio}")
            if self.targeted_classes:
                print(f"目标类别限制为: {self.targeted_classes}，样本翻转比例: {self.targeted_samples_ratio}")
            if self.dynamic_flip_schedule:
                print(f"使用动态翻转调度: {self.dynamic_flip_schedule}")
            if self.preserve_class_distribution:
                print("启用类别分布保持功能")

    def update_round(self, round_num):
        """更新当前轮次，用于动态翻转调度"""
        self.current_round = round_num
        if self.dynamic_flip_schedule:
            # 找到小于等于当前轮次的最大键
            applicable_rounds = [r for r in self.dynamic_flip_schedule.keys() if r <= round_num]
            if applicable_rounds:
                latest_round = max(applicable_rounds)
                self.flip_ratio = self.dynamic_flip_schedule[latest_round]
                print(f"轮次 {round_num}: 更新翻转比例为 {self.flip_ratio}")

    def get_effective_flip_ratio(self):
        """获取当前有效的翻转比例"""
        if self.dynamic_flip_schedule:
            # 找到小于等于当前轮次的最大键
            applicable_rounds = [r for r in self.dynamic_flip_schedule.keys() if r <= self.current_round]
            if applicable_rounds:
                latest_round = max(applicable_rounds)
                return self.dynamic_flip_schedule[latest_round]
        return self.flip_ratio

    def __call__(self, label):
        """对单个标签应用翻转逻辑。"""
        # 首先检查是否应该翻转这个标签（基于翻转比例）
        if random.random() > self.get_effective_flip_ratio():
            return label
            
        # 检查是否是目标类别
        if self.targeted_classes and label not in self.targeted_classes:
            return label
            
        # 对于目标类别，进一步根据样本翻转比例决定是否翻转
        if self.targeted_classes and random.random() > self.targeted_samples_ratio:
            return label
            
        if self.mapping.get("type") == "random_others":
            # 随机翻转到其他类别
            # 确保不会翻转到自身，且 NUM_CLASSES 至少有2个类别才能翻转
            if NUM_CLASSES <= 1:
                return label # 无法翻转到其他类别
            new_label = random.choice([i for i in range(NUM_CLASSES) if i != label])
            return new_label
        
        # 应用预定义的映射
        # 如果标签不在映射中，则返回原标签（不翻转）
        return self.mapping.get(label, label)
        
    def flip_dataset(self, labels):
        """翻转整个数据集的标签"""
        # 如果需要保持类别分布，先保存原始分布
        if self.preserve_class_distribution:
            unique_labels, counts = np.unique(labels, return_counts=True)
            self.original_class_distribution = dict(zip(unique_labels, counts))
            
            # 为每个类别确定应该翻转到哪些目标类别及数量
            for original_class in unique_labels:
                # 这个类别应该翻转的样本数
                num_to_flip = int(self.original_class_distribution[original_class] * self.get_effective_flip_ratio())
                if num_to_flip <= 0:
                    continue
                    
                # 根据映射确定目标类别
                if self.mapping.get("type") == "random_others":
                    # 随机选择多个目标类别
                    target_classes = [i for i in range(NUM_CLASSES) if i != original_class]
                    if not target_classes:
                        continue  # 没有可能的目标类别
                else:
                    # 使用预定义映射的目标类别
                    target_class = self.mapping.get(original_class)
                    if target_class is None or target_class == original_class:
                        continue  # 没有有效的目标类别
                    target_classes = [target_class]
                    
                # 为每个原始类别分配翻转目标及数量
                self.class_flip_targets[original_class] = {
                    "targets": target_classes,
                    "count": num_to_flip
                }
                
            # 执行保持分布的翻转
            labels_array = np.array(labels) if not isinstance(labels, np.ndarray) else labels.copy()
            
            for original_class, flip_info in self.class_flip_targets.items():
                # 找到该类别的所有样本索引
                indices = np.where(labels_array == original_class)[0]
                if len(indices) == 0:
                    continue
                    
                # 随机选择要翻转的样本
                num_to_flip = min(flip_info["count"], len(indices))
                flip_indices = np.random.choice(indices, size=num_to_flip, replace=False)
                
                # 将这些样本翻转到目标类别
                for idx in flip_indices:
                    target = random.choice(flip_info["targets"])
                    labels_array[idx] = target
                    
            flipped_labels = labels_array
            
        else:
            # 常规翻转（不保持分布）
            if isinstance(labels, np.ndarray):
                flipped_labels = np.array([self(label) for label in labels])
            else:
                flipped_labels = [self(label) for label in labels]
            
        # 计算实际翻转的标签数量和比例
        flipped_count = sum(1 for orig, flipped in zip(labels, flipped_labels) if orig != flipped)
        actual_ratio = flipped_count / len(labels) if len(labels) > 0 else 0
        
        print(f"标签翻转完成: 共翻转 {flipped_count}/{len(labels)} 个标签 (比例: {actual_ratio:.2f})")
        
        # 如果启用了类别分布保持，验证结果
        if self.preserve_class_distribution and self.original_class_distribution:
            flipped_unique, flipped_counts = np.unique(flipped_labels, return_counts=True)
            flipped_distribution = dict(zip(flipped_unique, flipped_counts))
            print(f"原始类别分布: {self.original_class_distribution}")
            print(f"翻转后类别分布: {flipped_distribution}")
            
        return flipped_labels

if __name__ == '__main__':
    print(f"使用的类别数 NUM_CLASSES = {NUM_CLASSES}\n")  # 中文打印

    # --- 测试高混淆翻转 ---
    flipper_high = LabelFlipper(flip_type="high_confusion", flip_ratio=0.7)
    print("\n--- 测试高混淆翻转 (70%比例) ---")
    test_labels_high = [0, 2, 9, 33, 19, 100] # 100 不在映射中，应保持不变
    for label in test_labels_high:
        flipped_label = flipper_high(label)
        print(f"原始标签: {label} -> 翻转后(高混淆): {flipped_label}")
    
    # --- 测试最大对抗性翻转 ---
    flipper_max_adv = LabelFlipper(flip_type="max_adversarial", flip_ratio=1.0)
    print("\n--- 测试最大对抗性翻转 (100%比例) ---")
    test_labels_adv = [0, 1, 2, 3, 4, 5]
    for label in test_labels_adv:
        flipped_label = flipper_max_adv(label)
        print(f"原始标签: {label} -> 翻转后(最大对抗): {flipped_label}")
        
    # --- 测试安全关键类别翻转 ---
    flipper_critical = LabelFlipper(flip_type="critical_safety", flip_ratio=1.0)
    print("\n--- 测试安全关键类别翻转 (100%比例) ---")
    test_labels_critical = [13, 14, 17, 18, 26, 27, 19, 20, 0, 1]  # 包括一些不在映射中的类别
    for label in test_labels_critical:
        flipped_label = flipper_critical(label)
        print(f"原始标签: {label} -> 翻转后(安全关键): {flipped_label}")

    # --- 测试完全随机翻转 ---
    flipper_complete_random = LabelFlipper(flip_type="completely_random", flip_ratio=1.0)
    print("\n--- 测试完全随机翻转 (100%比例) ---")
    test_labels_random = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    for label in test_labels_random:
        flipped_label = flipper_complete_random(label)
        print(f"原始标签: {label} -> 翻转后(完全随机): {flipped_label}")

    # --- 测试目标类别限制和翻转比例 ---
    flipper_targeted = LabelFlipper(
        flip_type="max_adversarial", 
        flip_ratio=1.0,
        targeted_classes=[0, 1, 2, 3, 4],  # 只翻转这些类别
        targeted_samples_ratio=0.7  # 目标类别中70%的样本被翻转
    )
    print("\n--- 测试目标类别和样本比例限制 ---")
    # 创建一个包含多个样本的测试集，每个类别多个样本
    test_labels_array = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6])
    flipped_labels = flipper_targeted.flip_dataset(test_labels_array)
    for orig, flipped in zip(test_labels_array, flipped_labels):
        print(f"原始: {orig} -> 翻转后: {flipped}")
        
    # --- 测试动态翻转调度 ---
    dynamic_flipper = LabelFlipper(
        flip_type="directed_confusion",
        flip_ratio=0.1,  # 初始翻转率
        dynamic_flip_schedule={0: 0.1, 5: 0.5, 10: 0.9}  # 动态调度
    )
    print("\n--- 测试动态翻转调度 ---")
    for round_num in [0, 3, 5, 7, 10, 15]:
        dynamic_flipper.update_round(round_num)
        print(f"轮次 {round_num}, 翻转率: {dynamic_flipper.get_effective_flip_ratio()}")
        
    # --- 测试类别分布保持 ---
    preserving_flipper = LabelFlipper(
        flip_type="directed_confusion",
        flip_ratio=0.5,
        preserve_class_distribution=True
    )
    print("\n--- 测试类别分布保持 ---")
    # 创建一个有明确分布的测试集
    distribution_test = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    print(f"原始分布: {np.unique(distribution_test, return_counts=True)}")
    flipped_with_preserved_dist = preserving_flipper.flip_dataset(distribution_test)
    print(f"翻转后: {np.unique(flipped_with_preserved_dist, return_counts=True)}")