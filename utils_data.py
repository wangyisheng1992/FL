# utils_data.py
import pandas as pd
from PIL import Image
import numpy as np
# train_test_split is removed from here as it's now handled in main.py before calling prepare_dataset_partitions
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from collections import Counter, defaultdict
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 设置黑体为默认字体
matplotlib.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
import matplotlib.pyplot as plt
import math
import os # 导入 os 模块，用于检查文件是否存在
import io

# 假设的类别数量 (需要根据您的数据集调整，例如德国交通标志识别基准 GTSRB)
NUM_CLASSES = 43
# 图像大小，这里调整为 112x112 以适应 Vision Transformer 模型
IMG_SIZE = 112

def load_dataset_from_parquet(file_path):
    """
    从 Parquet 文件加载数据集。
    Parquet 文件应包含 'image' (PIL.Image 对象) 和 'label' (整数标签) 列。
    """
    try:
        df = pd.read_parquet(file_path)
        print(f"数据集从 {file_path} 成功加载。形状: {df.shape}") # 中文打印
        # 确保 'image' 列包含 PIL.Image 对象，'label' 列包含整数标签
        # Check if the 'image' column contains dictionaries with 'bytes'
        if not df.empty and isinstance(df['image'].iloc[0], dict) and 'bytes' in df['image'].iloc[0]:
            print("检测到 'image' 列包含字典，将尝试解码 'bytes' 字段为 PIL Image。") # 中文打印
            def decode_image_if_needed(img_data):
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    return Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
                return img_data # Assume it's already a PIL Image otherwise
            df['image'] = df['image'].apply(decode_image_if_needed)

        if not df.empty and not all(isinstance(img, Image.Image) for img in df['image'].head()):
            print("警告: 'image' 列在解码后仍未完全包含预期的 PIL Image 对象。") # 中文打印
        return df
    except Exception as e:
        print(f"加载 Parquet 文件 {file_path} 时出错: {e}") # 中文打印
        print("创建一个虚拟 DataFrame 用于演示。") # 中文打印
        num_dummy_samples = 100
        dummy_images = [Image.new('RGB', (IMG_SIZE, IMG_SIZE), color = 'red') for _ in range(num_dummy_samples)]
        dummy_labels = np.random.randint(0, NUM_CLASSES, num_dummy_samples)
        return pd.DataFrame({'image': dummy_images, 'label': dummy_labels})


class TrafficSignDataset(Dataset):
    """自定义交通标志数据集类。"""
    def __init__(self, dataframe, transform=None, target_transform=None):
        self.dataframe = dataframe
        self.transform = transform # 图像转换，如缩放、归一化
        self.target_transform = target_transform # 标签转换，用于标签翻转等数据增强

    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        根据索引获取一个样本及其标签。
        Args:
            idx (int): 样本的索引。
        Returns:
            tuple: (image, label) 包含图像和对应的标签。
        """
        image_data = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['label']

        # Ensure image is a PIL Image object
        image = image_data
        if isinstance(image_data, dict) and 'bytes' in image_data: # Should have been handled by load_dataset_from_parquet
            image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        elif not isinstance(image_data, Image.Image):
            # Fallback or error if not already a PIL image (e.g. if it's still raw bytes)
            # This indicates an issue upstream if images are not decoded properly before this stage
            raise TypeError(f"Unexpected image data type: {type(image_data)}. Expected PIL Image.")

        # 在应用任何外部 transform 之前，强制将图像大小调整为 IMG_SIZE x IMG_SIZE
        image = transforms.Resize((IMG_SIZE, IMG_SIZE))(image)

        if self.transform:
            image = self.transform(image)
        if self.target_transform: # 如果提供了标签转换 (例如用于标签翻转攻击)
            label = self.target_transform(label)

        return image, label

def get_transforms(enable_augmentation=False):
    """
    获取图像预处理转换。
    Args:
        enable_augmentation (bool): 是否启用数据增强（如随机旋转、水平翻转）。
    Returns:
        torchvision.transforms.Compose: 包含一系列图像转换的组合对象。
    """
    transform_list = []
    if enable_augmentation:
        augmentation_transforms = [
            transforms.RandomRotation(degrees=15), # 随机旋转15度
            transforms.RandomHorizontalFlip(p=0.5), # 50%概率随机水平翻转
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 可选: 颜色抖动
        ]
        transform_list.extend(augmentation_transforms)

    transform_list.extend([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # 调整图像大小
        transforms.ToTensor(), # 将 PIL Image 或 NumPy 数组转换为 Tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 标准化到 [-1, 1] 范围
    ])
    return transforms.Compose(transform_list)

def prepare_dataset_partitions(train_df_for_clients, num_clients, is_iid=True, dirichlet_alpha=0.5):
    if train_df_for_clients.empty:
        print("错误：提供给客户端分区的训练数据为空。")
        # Return a list of empty DataFrames of length num_clients
        return [pd.DataFrame() for _ in range(num_clients)], defaultdict(lambda: defaultdict(int))

    print(f"开始为 {num_clients} 个客户端准备数据分区，源数据大小: {len(train_df_for_clients)}")

    # Initialize a list of empty DataFrames, one for each client
    client_datasets = [pd.DataFrame() for _ in range(num_clients)]
    # client_data_indices will store lists of df indices for each client
    client_data_indices = [[] for _ in range(num_clients)]
    client_class_counts = defaultdict(lambda: defaultdict(int))

    if is_iid:
        print("进行 IID 数据划分...")
        shuffled_indices = train_df_for_clients.index.to_numpy()
        np.random.shuffle(shuffled_indices)
        split_indices_list = np.array_split(shuffled_indices, num_clients)
        for i in range(num_clients):
            client_data_indices[i] = split_indices_list[i].tolist()
            if client_data_indices[i]: # Only if there are indices
                 client_datasets[i] = train_df_for_clients.loc[client_data_indices[i]]
                 for label in client_datasets[i]['label']:
                    client_class_counts[i][int(label)] += 1
            # If client_data_indices[i] is empty, client_datasets[i] remains an empty DataFrame
    else:
        print(f"进行 Non-IID 数据划分 (Dirichlet alpha={dirichlet_alpha})...")
        labels = train_df_for_clients['label'].astype(int).to_numpy()
        min_label_val, max_label_val = labels.min(), labels.max()
        
        # Ensure num_classes_present covers all possible labels from 0 to max_label_val
        # If dataset labels are not contiguous from 0, this might need adjustment
        # or use NUM_CLASSES if labels are guaranteed to be within that range.
        # For GTSRB, labels are 0 to 42.
        num_classes_present_in_data = max_label_val + 1 
        
        # Use NUM_CLASSES for dirichlet distribution if it's the global constant for max classes
        # This ensures the distribution has a slot for every possible class, even if not in this specific train_df_for_clients
        actual_num_classes_for_dirichlet = NUM_CLASSES 

        # label_distribution will be (num_clients, actual_num_classes_for_dirichlet)
        label_distribution = np.random.dirichlet([dirichlet_alpha] * actual_num_classes_for_dirichlet, num_clients)

        class_indices_map = {
            cls_label: train_df_for_clients.index[labels == cls_label].tolist()
            for cls_label in range(actual_num_classes_for_dirichlet) # Iterate up to global NUM_CLASSES
                                                                  # or num_classes_present_in_data if labels are compact
        }

        # For each class, distribute its samples to clients
        for cls_label in range(actual_num_classes_for_dirichlet):
            if not class_indices_map.get(cls_label): # Use .get() for safety if a class is missing
                continue

            df_indices_for_class_k = class_indices_map[cls_label]
            np.random.shuffle(df_indices_for_class_k)
            total_samples_in_class_k = len(df_indices_for_class_k)
            
            if total_samples_in_class_k == 0:
                continue

            proportions_for_class_k_on_clients = label_distribution[:, cls_label]
            num_samples_per_client_for_class_k = \
                (proportions_for_class_k_on_clients * total_samples_in_class_k).astype(int)
            
            remainder = total_samples_in_class_k - np.sum(num_samples_per_client_for_class_k)
            if remainder > 0:
                sorted_client_indices_by_prop = np.argsort(-proportions_for_class_k_on_clients)
                for r_idx in range(remainder):
                    num_samples_per_client_for_class_k[sorted_client_indices_by_prop[r_idx % num_clients]] += 1

            current_pos_in_class_k_indices = 0
            for client_idx in range(num_clients):
                num_to_assign = num_samples_per_client_for_class_k[client_idx]
                assigned_df_indices = df_indices_for_class_k[
                    current_pos_in_class_k_indices : current_pos_in_class_k_indices + num_to_assign
                ]
                client_data_indices[client_idx].extend(assigned_df_indices) # Collect all indices for a client
                # client_class_counts will be built later from the final client_data_indices
                current_pos_in_class_k_indices += num_to_assign
        
        # Create DataFrames from collected indices and count classes
        for client_idx in range(num_clients):
            if client_data_indices[client_idx]: # If client has any data indices
                np.random.shuffle(client_data_indices[client_idx]) # Shuffle indices for this client
                client_datasets[client_idx] = train_df_for_clients.loc[client_data_indices[client_idx]]
                for label_val in client_datasets[client_idx]['label']:
                    client_class_counts[client_idx][int(label_val)] += 1
            # If client_data_indices[client_idx] is empty, client_datasets[client_idx] remains an empty DataFrame

    # Ensure the returned list `client_datasets` always has `num_clients` elements.
    # The initialization `client_datasets = [pd.DataFrame() for _ in range(num_clients)]` handles this.
    actual_clients_with_data = sum(1 for df in client_datasets if not df.empty)
    if actual_clients_with_data < num_clients:
         print(f"警告: {num_clients - actual_clients_with_data} 个客户端没有分配到数据。这在Non-IID场景下可能发生。")

    return client_datasets, client_class_counts


def get_dataloader(df, batch_size, shuffle=True, enable_augmentation=False, enable_balanced_sampling=False, target_transform=None):
    """
    为给定的 DataFrame 创建 DataLoader。

    Args:
        df (pd.DataFrame): 要创建 DataLoader 的 DataFrame。
        batch_size (int): 批次大小。
        shuffle (bool): 是否打乱数据。
        enable_augmentation (bool): 是否启用数据增强。
        enable_balanced_sampling (bool): 是否启用加权随机采样以平衡类别。
        target_transform (callable, optional): 应用于标签的转换函数。
    Returns:
        torch.utils.data.DataLoader: 创建好的 DataLoader 对象。
    """
    if df.empty:
        print("警告: 尝试为Kong DataFrame 创建 DataLoader。将返回一个空的 DataLoader。") # 中文打印
        return DataLoader(TrafficSignDataset(df), batch_size=batch_size, shuffle=False)


    transform = get_transforms(enable_augmentation)
    dataset = TrafficSignDataset(df, transform=transform, target_transform=target_transform)

    sampler = None
    # Ensure 'label' column exists for balanced sampling
    if shuffle and enable_balanced_sampling and 'label' in df.columns and not df['label'].empty:
        try:
            labels = df['label'].tolist() # Directly use labels from DataFrame
            class_counts = Counter(labels)

            if not class_counts:
                print("警告: 数据集标签为空，无法进行平衡采样。将使用标准 DataLoader。") # 中文打印
                return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

            weights_per_class = {cls: 1.0 / count for cls, count in class_counts.items() if count > 0}
            sample_weights = [weights_per_class.get(label, 0) for label in labels] # Use labels from df

            if sum(sample_weights) > 0:
                sampler = WeightedRandomSampler(
                    weights=torch.DoubleTensor(sample_weights),
                    num_samples=len(sample_weights),
                    replacement=True
                )
                shuffle = False # Sampler handles shuffling
            else:
                print("警告: 无法创建 WeightedRandomSampler，总权重为零。将使用标准打乱模式。") # 中文打印
        except Exception as e:
            print(f"平衡采样出错: {e}。将使用标准打乱模式。") # 中文打印
    elif enable_balanced_sampling and ('label' not in df.columns or df['label'].empty):
        print("警告: 请求了平衡采样，但 'label' 列缺失或为空。将使用标准打乱模式。") # 中文打印


    # num_workers=0 表示在主进程中加载数据
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0)

def visualize_client_data_distribution(client_class_counts, num_clients_to_show=10, title="客户端数据分布", save_path=None):
    """
    可视化指定数量客户端的数据类别分布。

    Args:
        client_class_counts (defaultdict): 每个客户端的类别计数，格式为 {client_id: {class_id: count}}.
        num_clients_to_show (int): 要显示的客户端数量。
        title (str): 图表标题。
        save_path (str, optional): 保存图表的路径。如果为 None，则显示图表。
    """
    if not client_class_counts:
        print("没有客户端类别计数数据可供可视化。")
        return

    # Filter to get only clients that actually have data for visualization
    clients_with_data_ids = sorted([
        cid for cid, counts in client_class_counts.items() if sum(counts.values()) > 0
    ])

    if not clients_with_data_ids:
        print("在所有客户端中未找到有效的类别计数数据，无法可视化。")
        return
        
    display_count_actual = min(len(clients_with_data_ids), num_clients_to_show)
    
    if display_count_actual == 0:
        print("没有可供显示的拥有数据的客户端。")
        return
        
    client_ids_to_show = clients_with_data_ids[:display_count_actual]

    all_classes_set = set()
    for cid in client_ids_to_show:
        if cid in client_class_counts: # Should always be true due to pre-filtering
            for cls_label in client_class_counts[cid]:
                all_classes_set.add(cls_label)
    
    if not all_classes_set:
        print("在选定的客户端中未找到类别数据。")
        return
            
    all_classes = sorted(list(all_classes_set))

    plot_data = {}
    for cls in all_classes:
        plot_data[cls] = [client_class_counts.get(cid, {}).get(cls, 0) for cid in client_ids_to_show]

    # Dynamically adjust figure width based on the number of clients
    # Dynamically adjust figure height based on number of classes for legend
    base_fig_width = 10
    fig_width = max(base_fig_width, display_count_actual * 1.0) # Increase width per client shown
    
    # Adjust height based on number of legend items
    num_legend_items = len(all_classes)
    fig_height = 7 + max(0, (num_legend_items - 20) * 0.15) # Add height for long legends

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=110) 

    bottom = np.zeros(display_count_actual)
    
    # Use a consistent colormap
    # Using 'viridis', 'plasma', 'cividis', or 'tab20' for many categories
    if len(all_classes) <= 20:
        colors = plt.cm.get_cmap('tab20', len(all_classes))
    else: # Fallback for more than 20 classes, might repeat colors but still better
        colors = plt.cm.get_cmap('nipy_spectral', len(all_classes))


    for i, (cls, counts_for_cls) in enumerate(plot_data.items()):
        ax.bar(
            [f'客户端 {cid}' for cid in client_ids_to_show],
            counts_for_cls,
            label=f'类别 {cls}',
            bottom=bottom,
            color=colors(i) if len(all_classes) > 1 else colors(0) # Handle single class case for colormap
        )
        bottom += np.array(counts_for_cls)

    ax.set_xlabel("客户端 ID", fontsize=12) 
    ax.set_ylabel("样本数量", fontsize=12) 
    ax.set_title(title, fontsize=14, pad=20) # Add padding to title
    
    # Legend handling: more columns if many items, smaller font
    num_legend_cols = 1
    legend_fontsize = 9
    if num_legend_items > 30:
        num_legend_cols = 3
        legend_fontsize = 7
    elif num_legend_items > 15:
        num_legend_cols = 2
        legend_fontsize = 8

    # Position legend to the right, adjust subplot to make space
    leg = ax.legend(title="类别", bbox_to_anchor=(1.03, 1), loc='upper left', fontsize=legend_fontsize, ncol=num_legend_cols)
    
    # Calculate how much to shrink the plot area based on legend width
    # This is an approximation and might need fine-tuning
    # Or rely more on bbox_inches='tight' during savefig.
    # For a more robust approach, we draw the legend first to get its width, then adjust.
    # However, a simpler adjustment with subplots_adjust often works.
    
    # Determine the right boundary for subplots_adjust
    # This value might need to be tuned based on your typical number of legend entries
    # A smaller value (e.g., 0.70) means more space for the legend.
    if num_legend_cols == 1:
        right_boundary = 0.82
    elif num_legend_cols == 2:
        right_boundary = 0.70
    else: # 3 columns
        right_boundary = 0.60 
        if fig_width < 15 : # if figure is not very wide, need even more relative space
            right_boundary = 0.55


    fig.subplots_adjust(left=0.1, bottom=0.15, right=right_boundary, top=0.9)


    plt.xticks(rotation=45, ha="right", fontsize=10) 
    plt.yticks(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y') # Only y-axis grid for clarity
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight') # Crucial for fitting everything
        print(f"数据分布图已保存到 {save_path}") 
    else:
        plt.show()
    plt.close(fig)

def partition_noniid_custom(train_df, num_clients, num_rich_clients=2, num_classes_per_poor_client=1, rich_client_class_ratio=0.8):
    """
    自定义极端Non-IID划分：部分客户端拥有大部分类别，其余客户端只拥有极少类别。
    Args:
        train_df: 完整数据集DataFrame，需有'label'列。
        num_clients: 客户端总数。
        num_rich_clients: 拥有多类别的客户端数量。
        num_classes_per_poor_client: 其余客户端拥有的类别数。
        rich_client_class_ratio: 富有客户端拥有的类别比例（如0.8表示80%的类别）。
    Returns:
        client_datasets, client_class_counts
    """
    labels = sorted(train_df['label'].unique())
    num_classes = len(labels)
    client_datasets = [pd.DataFrame() for _ in range(num_clients)]
    client_class_counts = defaultdict(lambda: defaultdict(int))

    # 随机选出富有客户端
    rich_client_indices = np.random.choice(range(num_clients), num_rich_clients, replace=False)
    poor_client_indices = [i for i in range(num_clients) if i not in rich_client_indices]

    # 富有客户端拥有的类别
    num_rich_classes = max(1, int(num_classes * rich_client_class_ratio))
    rich_classes = np.random.choice(labels, num_rich_classes, replace=False)

    # 给富有客户端分配大部分类别
    for idx in rich_client_indices:
        client_df = train_df[train_df['label'].isin(rich_classes)].copy()
        client_datasets[idx] = client_df
        for label in client_df['label']:
            client_class_counts[idx][int(label)] += 1

    # 给贫穷客户端分配极少类别
    for idx in poor_client_indices:
        poor_classes = np.random.choice(labels, num_classes_per_poor_client, replace=False)
        client_df = train_df[train_df['label'].isin(poor_classes)].copy()
        client_datasets[idx] = client_df
        for label in client_df['label']:
            client_class_counts[idx][int(label)] += 1

    return client_datasets, client_class_counts

if __name__ == '__main__':
    # 示例用法
    dummy_train_parquet_path = r'E:\Download\fl_clas\fl_class\data\dummy_train_data.parquet' # 虚拟训练文件路径
    if dummy_train_parquet_path:
        # 1. 首先加载完整的数据集 (模拟在 main.py 中的操作)
        full_train_data_df = load_dataset_from_parquet(dummy_train_parquet_path)

        if not full_train_data_df.empty:
            print("\n--- 测试 IID 数据划分 ---") # 中文打印
            client_dfs_iid, client_class_counts_iid = prepare_dataset_partitions(
                full_train_data_df, num_clients=10, is_iid=True
            )
            if client_dfs_iid:
                print(f"IID 客户端数量: {len(client_dfs_iid)}") # 中文打印
                if client_dfs_iid[0] is not None and not client_dfs_iid[0].empty:
                    print(f"第一个 IID 客户端的数据点数量: {len(client_dfs_iid[0])}") # 中文打印
                    dl_iid = get_dataloader(client_dfs_iid[0], batch_size=4, enable_augmentation=True, enable_balanced_sampling=True)
                    for x, y in dl_iid:
                        print(f"IID 批次形状: {x.shape}，标签: {y}") # 中文打印
                        break
                visualize_client_data_distribution(client_class_counts_iid, num_clients_to_show=5, title="IID 数据分布测试") # 中文打印

            print("\n--- 测试自定义极端 Non-IID 数据划分 ---") # 中文打印
            client_dfs_custom, client_class_counts_custom = partition_noniid_custom(
                full_train_data_df, num_clients=10, num_rich_clients=1, num_classes_per_poor_client=1, rich_client_class_ratio=0.8
            )
            if client_dfs_custom:
                print(f"自定义 Non-IID 客户端数量: {len(client_dfs_custom)}") # 中文打印
                if client_dfs_custom[0] is not None and not client_dfs_custom[0].empty:
                    print(f"第一个自定义 Non-IID 客户端的数据点数量: {len(client_dfs_custom[0])}") # 中文打印
                    dl_custom = get_dataloader(client_dfs_custom[0], batch_size=4)
                    for x, y in dl_custom:
                        print(f"自定义 Non-IID 批次形状: {x.shape}，标签: {y}") # 中文打印
                        break
                visualize_client_data_distribution(client_class_counts_custom, num_clients_to_show=10, title="自定义极端 Non-IID 数据分布", save_path="fl_results_20250520_163934/custom_noniid_distribution.png") # 中文打印
        else:
            print("无法加载用于测试的虚拟训练数据。") # 中文打印
    else:
        print("跳过 utils_data.py 的 __main__ 测试，因为虚拟 Parquet 文件未创建。") # 中文打印