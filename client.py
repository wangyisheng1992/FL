# client.py
import flwr as fl
import torch
import torch.nn as nn
from torch.optim import AdamW
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

from model import get_model # 从 model.py 导入模型创建函数
from utils_data import get_dataloader, load_dataset_from_parquet # 从 utils_data.py 导入数据加载器函数
from attacks import LabelFlipper # 从 attacks.py 导入标签翻转器

# *** REMOVE 'from flwr.common import Context' if it was added here and client_fn won't use it directly ***

# 全局变量或从配置中读取设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 43 

class FlowerClient(fl.client.NumPyClient):
    """
    一个自定义的 Flower 客户端，负责在本地训练和评估模型，并支持模拟攻击。
    """
    def __init__(self, client_id, train_df, val_df, client_config):
        self.client_id = client_id
        self.train_df = train_df
        self.val_df = val_df
        self.config = client_config

        # 关键：根据 config 里的 use_vit_base_config 初始化模型
        self.model = get_model(use_vit_base_config=self.config.get("use_vit_base_config", False)).to(DEVICE)
        self.is_attacker = False
        self.label_flipper = None
        
        # 添加防御相关的属性
        self.enable_model_validation = self.config.get("enable_model_validation", False)
        self.enable_anomaly_detection = self.config.get("enable_anomaly_detection", False)
        self.enable_gradient_clipping = self.config.get("gradient_clipping", False)
        self.gradient_clip_norm = self.config.get("clip_norm", 1.0)
        self.model_history = []  # 用于存储历史模型参数
        self.gradient_history = []  # 用于存储历史梯度
        
        # 新增防御机制
        self.enable_noise_injection = self.config.get("noise_scale", 0.0) > 0
        self.noise_scale = self.config.get("noise_scale", 0.01)
        self.enable_label_smoothing = self.config.get("label_smoothing", False)
        self.smoothing_factor = self.config.get("smoothing_factor", 0.1)
        self.enable_adaptive_lr = self.config.get("adaptive_learning_rate", False)
        self.min_lr = self.config.get("min_lr", 1e-5)
        self.max_lr = self.config.get("max_lr", 1e-3)
        self.lr_history = []  # 用于存储历史学习率

        if self.config.get("is_attacker", False):
            self.is_attacker = True
            attack_params = self.config.get("attack_params", {})
            flip_type = attack_params.get("label_flip_type", "random_others")
            flip_ratio = attack_params.get("flip_ratio", 1.0)
            targeted_classes = attack_params.get("targeted_classes", None)
            targeted_samples_ratio = attack_params.get("targeted_samples_ratio", 1.0)
            dynamic_flip_schedule = attack_params.get("dynamic_flip_schedule", None)
            preserve_class_distribution = attack_params.get("preserve_class_distribution", False)
            
            self.label_flipper = LabelFlipper(
                flip_type=flip_type,
                flip_ratio=flip_ratio,
                targeted_classes=targeted_classes,
                targeted_samples_ratio=targeted_samples_ratio,
                dynamic_flip_schedule=dynamic_flip_schedule,
                preserve_class_distribution=preserve_class_distribution
            )
            print(f"客户端 {self.client_id} 是一个攻击者，使用标签翻转类型: {flip_type}，翻转比例: {flip_ratio}。")
        else:
            print(f"客户端 {self.client_id} 是一个正常客户端。")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def _validate_model(self, model_params):
        """验证模型参数是否异常"""
        if not self.enable_model_validation:
            return True
            
        if not self.model_history:
            self.model_history.append(model_params)
            return True
            
        # 计算与历史模型的参数差异
        param_diff = np.mean([np.abs(p1 - p2).mean() for p1, p2 in zip(model_params, self.model_history[-1])])
        threshold = np.mean([np.abs(p).mean() for p in model_params]) * 0.1  # 使用参数平均值的10%作为阈值
        
        # 更新历史记录
        self.model_history.append(model_params)
        if len(self.model_history) > 5:  # 只保留最近5个模型
            self.model_history.pop(0)
            
        return param_diff < threshold

    def _detect_anomaly(self, gradients):
        """检测梯度异常"""
        if not self.enable_anomaly_detection:
            return False
            
        if not self.gradient_history:
            self.gradient_history.append(gradients)
            return False
            
        # 计算当前梯度的统计特征
        current_norm = np.mean([np.linalg.norm(g) for g in gradients])
        current_mean = np.mean([np.mean(g) for g in gradients])
        current_std = np.mean([np.std(g) for g in gradients])
        
        # 计算历史梯度的统计特征
        history_norms = [np.mean([np.linalg.norm(g) for g in gs]) for gs in self.gradient_history]
        history_means = [np.mean([np.mean(g) for g in gs]) for gs in self.gradient_history]
        history_stds = [np.mean([np.std(g) for g in gs]) for gs in self.gradient_history]
        
        # 计算与历史统计量的差异
        norm_diff = abs(current_norm - np.mean(history_norms)) / (np.std(history_norms) + 1e-10)
        mean_diff = abs(current_mean - np.mean(history_means)) / (np.std(history_means) + 1e-10)
        std_diff = abs(current_std - np.mean(history_stds)) / (np.std(history_stds) + 1e-10)
        
        # 更新历史记录
        self.gradient_history.append(gradients)
        if len(self.gradient_history) > 5:  # 只保留最近5个梯度
            self.gradient_history.pop(0)
            
        # 如果任一统计量差异超过3个标准差，认为是异常
        return norm_diff > 3 or mean_diff > 3 or std_diff > 3

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = self.config.get("local_epochs", 3)
        batch_size = self.config.get("batch_size", 16)
        learning_rate = self.config.get("learning_rate", 1e-4)
        enable_augmentation = self.config.get("enable_augmentation", False)
        enable_balanced_sampling = self.config.get("enable_balanced_sampling", False)
        target_transform_for_loader = self.label_flipper if self.is_attacker else None

        # 如果启用自适应学习率，根据历史性能调整学习率
        if self.enable_adaptive_lr and self.lr_history:
            last_loss = self.lr_history[-1]
            if last_loss > 0:
                # 如果损失增加，降低学习率；如果损失减少，增加学习率
                learning_rate = max(self.min_lr, min(self.max_lr, learning_rate * (0.9 if last_loss > self.lr_history[-2] else 1.1)))

        train_loader = get_dataloader(
            self.train_df,
            batch_size=batch_size,
            shuffle=True,
            enable_augmentation=enable_augmentation,
            enable_balanced_sampling=enable_balanced_sampling,
            target_transform=target_transform_for_loader
        )

        num_train_samples = len(train_loader.dataset) if train_loader.dataset else 0
        num_batches = len(train_loader)

        if num_batches == 0:
            return self.get_parameters(config={}), 0, {"loss": 0.0}

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # 使用标签平滑的损失函数
        if self.enable_label_smoothing:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.smoothing_factor)
        else:
            criterion = nn.CrossEntropyLoss()
            
        self.model.train()
        running_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            with tqdm(train_loader, desc=f"客户端 {self.client_id} Epoch {epoch+1}/{epochs}", leave=False, mininterval=0.1, miniters=max(1, num_batches // 100) if num_batches > 100 else 1) as tbar:
                for images, labels in tbar:
                    images, labels = images.to(DEVICE), labels.to(DEVICE).long()
                    
                    # 添加噪声注入
                    if self.enable_noise_injection:
                        noise = torch.randn_like(images) * self.noise_scale
                        images = images + noise
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # 获取当前梯度
                    current_gradients = [p.grad.clone().cpu().numpy() for p in self.model.parameters() if p.grad is not None]
                    
                    # 检测梯度异常
                    if self._detect_anomaly(current_gradients):
                        print(f"客户端 {self.client_id}: 检测到梯度异常，跳过此批次更新")
                        optimizer.zero_grad()
                        continue
                    
                    # 应用梯度裁剪
                    if self.enable_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    
                    optimizer.step()
                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    tbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        # 更新学习率历史
        if self.enable_adaptive_lr:
            self.lr_history.append(running_loss / (num_batches * epochs))
            if len(self.lr_history) > 5:  # 只保留最近5个损失值
                self.lr_history.pop(0)

        # 获取训练后的模型参数
        trained_params = self.get_parameters(config={})
        
        # 验证模型参数
        if not self._validate_model(trained_params):
            print(f"客户端 {self.client_id}: 模型参数验证失败，使用原始参数")
            return parameters, num_train_samples, {"loss": running_loss / (num_batches * epochs)}

        num_examples_fit = num_train_samples
        avg_fit_loss = (running_loss / (num_batches * epochs)) if (num_batches * epochs > 0) else 0.0
        return trained_params, num_examples_fit, {"loss": avg_fit_loss}

    def evaluate(self, parameters, config):
        if self.val_df.empty:
            # print(f"客户端 {self.client_id} 没有验证数据。跳过本地评估。")
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}
        self.set_parameters(parameters)
        batch_size = self.config.get("batch_size", 32)
        val_loader = get_dataloader(self.val_df, batch_size=batch_size, shuffle=False)

        if len(val_loader) == 0:
            # print(f"客户端 {self.client_id} 验证数据加载器为空。跳过本地评估。")
            return 0.0, 0, {"accuracy": 0.0, "loss": 0.0}

        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).long()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        return avg_loss, total, {"accuracy": accuracy, "loss": avg_loss}

# *** REVERT client_fn TO TAKE 'cid' AS THE FIRST ARGUMENT, NOT 'context' ***
def client_fn(cid: str, all_train_dfs, val_df, client_config_base, attacker_cids, attack_params, defense_params):
    """
    创建 FlowerClient 实例的工厂函数.
    Args:
        cid (str): 客户端的字符串ID.
        all_train_dfs (List[pd.DataFrame]): ...
        (rest of args)
    """
    client_id = int(cid) # cid is passed directly as a string

    print("client_id:", client_id, type(client_id), "all_train_dfs len:", len(all_train_dfs))

    client_train_df = all_train_dfs[int(client_id)]
    current_client_config = client_config_base.copy()
    current_client_config["enable_augmentation"] = defense_params.get("enable_augmentation", False)
    current_client_config["enable_balanced_sampling"] = defense_params.get("enable_balanced_sampling", False)

    if client_id in attacker_cids:
        current_client_config["is_attacker"] = True
        current_client_config["attack_params"] = attack_params
        # print(f"工厂函数: 客户端 {client_id} 被标记为攻击者。")
    else:
        current_client_config["is_attacker"] = False
        # print(f"工厂函数: 客户端 {client_id} 被标记为正常客户端。")

    return FlowerClient(client_id, client_train_df, val_df, current_client_config).to_client()

if __name__ == '__main__':
    from utils_data import prepare_dataset_partitions, NUM_CLASSES as DATA_NUM_CLASSES
    assert NUM_CLASSES == DATA_NUM_CLASSES, "NUM_CLASSES 在 client.py 和 utils_data.py 之间不匹配！"
    
    # dummy_parquet = r'E:\Download\fl_clas\fl_class\data\train-00000-of-00001.parquet' 
    # Use a relative path for testing in potentially different environments
    dummy_parquet = 'dummy_train_data.parquet' # Ensure this file exists or is created by utils_data.py's main
    
    import os
    # Create a dummy parquet if it doesn't exist for the test
    if not os.path.exists(dummy_parquet):
        print(f"Creating dummy Parquet file {dummy_parquet} for testing client.py...")
        from PIL import Image
        import pandas as pd
        import numpy as np
        num_dummy_samples = 250 # Enough to avoid issues with splitting/batching
        dummy_images = [{"bytes": Image.new('RGB', (224, 224), color = np.random.choice(['red', 'green', 'blue'])).tobytes() } for _ in range(num_dummy_samples)]
        dummy_labels = np.random.randint(0, NUM_CLASSES, num_dummy_samples)
        # Ensure enough samples per class for potential stratification if utils_data.py were to do it
        for i in range(NUM_CLASSES):
            if i < len(dummy_labels): # Make sure at least one of each class if possible
                dummy_labels[i] = i
        
        # Create DataFrame with 'image' as dicts with 'bytes' key, similar to Hugging Face datasets
        # For PIL Image objects directly, it would be:
        # dummy_images_pil = [Image.new('RGB', (224, 224), color = np.random.choice(['red', 'green', 'blue'])) for _ in range(num_dummy_samples)]
        # pd.DataFrame({'image': dummy_images_pil, 'label': dummy_labels}).to_parquet(dummy_parquet)

        # Simulating Hugging Face Dataset structure where 'image' column might contain PIL.Image objects directly after .with_format("pandas")
        # Or it might be dicts that need decoding. load_dataset_from_parquet handles the dict case.
        # For simplicity, let's create it with PIL Image objects directly for this dummy file.
        dummy_images_pil = [Image.new('RGB', (224, 224), color = np.random.choice(['red', 'green', 'blue'])) for _ in range(num_dummy_samples)]
        # The 'image' column in parquet from Hugging Face datasets with_format="pandas" often stores raw PIL.Image objects
        # which pandas can handle with pyarrow. Pandas itself doesn't natively store PIL.Image in parquet without workarounds.
        # For this test, we'll rely on load_dataset_from_parquet to create a DataFrame with PIL.Image if the file is missing.
        # The creation part in utils_data.py for load_dataset_from_parquet creates PIL.Image objects.
        # To be more robust, we can just let load_dataset_from_parquet create the dummy data if the file is not found.
        # Forcing the creation of a parquet file here might be too complex for the test setup if pyarrow has issues.
        # Let's assume utils_data.py 's main block or load_dataset_from_parquet's fallback creates usable data.
        # The important part is that `prepare_dataset_partitions` gets a valid DataFrame.

    # Assuming dummy_parquet path is correct or load_dataset_from_parquet handles its absence by creating dummy data
    # The `prepare_dataset_partitions` in `utils_data.py` loads this.
    # For the client.py unit test, we need to simulate what main.py does:
    # 1. Load data (or use dummy)
    # 2. Partition it for clients

    # This part tries to load/generate data via utils_data.py's functions
    # This is more of an integration test for prepare_dataset_partitions if dummy_parquet is complex.
    # For a simpler unit test of client_fn, we can mock the data directly.
    
    print(f"Attempting to load or generate data using: {dummy_parquet}")
    # This will use the fallback in load_dataset_from_parquet if the file doesn't exist
    raw_df_for_test = load_dataset_from_parquet(dummy_parquet) 

    if raw_df_for_test.empty or len(raw_df_for_test) < 10: # Check if enough data was loaded/generated
        print(f"Warning: Dummy data from {dummy_parquet} is empty or too small. Creating more explicit dummy data for client.py test.")
        from PIL import Image
        import pandas as pd
        import numpy as np
        num_dummy_samples = 250
        # Create PIL Image objects directly
        dummy_images_pil = [Image.new('RGB', (224, 224), color = np.random.choice(['red', 'green', 'blue'])) for _ in range(num_dummy_samples)]
        dummy_labels = np.random.randint(0, NUM_CLASSES, num_dummy_samples)
        for i in range(NUM_CLASSES): # Ensure at least one of each class for stability in splits
             if i < len(dummy_labels): dummy_labels[i] = i
        raw_df_for_test = pd.DataFrame({'image': dummy_images_pil, 'label': dummy_labels})

    # Now, partition this data for clients (mimicking main.py's data prep for clients)
    # Use a small number of clients for the test.
    client_dfs_test, _, _ = prepare_dataset_partitions(raw_df_for_test, num_clients=2, is_iid=True) 
    
    # Use a dummy test_df for FlowerClient's val_df argument, as it expects one
    # (even if it's empty and evaluation is skipped)
    # Create a small dummy validation set
    dummy_val_images_pil = [Image.new('RGB', (224, 224), color = 'yellow') for _ in range(10)]
    dummy_val_labels = np.random.randint(0, NUM_CLASSES, 10)
    dummy_test_df_for_client = pd.DataFrame({'image': dummy_val_images_pil, 'label': dummy_val_labels})


    if client_dfs_test and client_dfs_test[0] is not None and not client_dfs_test[0].empty:
        cfg_normal_test = {"local_epochs": 1, "batch_size": 2, "learning_rate": 1e-3} # Small epochs/batch for speed
        
        # Call client_fn directly with string CID and other args for testing
        client0_numpy_client = client_fn(
            cid='0', # Pass CID as string
            all_train_dfs=client_dfs_test, # List of DFs for all test clients
            val_df=dummy_test_df_for_client,    # Dummy validation data for this client
            client_config_base=cfg_normal_test,
            attacker_cids=[],
            attack_params={},
            defense_params={}
        ).to_numpy_client() # Convert to NumPyClient for fit/evaluate calls

        print(f"\n--- 正在测试正常客户端 {client0_numpy_client.client_id} ---")
        initial_params = client0_numpy_client.get_parameters(config={})
        trained_params, num_examples, metrics = client0_numpy_client.fit(initial_params, config={})
        print(f"训练完成: {num_examples} 个样本，指标: {metrics}")
        loss, num_eval_examples, eval_metrics = client0_numpy_client.evaluate(trained_params, config={})
        print(f"评估完成: 损失 {loss:.4f}，{num_eval_examples} 个样本，指标: {eval_metrics}")

        if len(client_dfs_test) > 1 and client_dfs_test[1] is not None and not client_dfs_test[1].empty:
            cfg_attacker_test = {
                "local_epochs": 1, "batch_size": 2, "learning_rate": 1e-3,
                "is_attacker": True,
                "attack_params": {"label_flip_type": "high_confusion"}
            }
            client1_numpy_client = client_fn(
                cid='1', # Pass CID as string
                all_train_dfs=client_dfs_test,
                val_df=dummy_test_df_for_client,
                client_config_base=cfg_attacker_test, # Base config contains attacker flag
                attacker_cids=[1], # Explicitly pass the list of attacker cids
                attack_params=cfg_attacker_test["attack_params"], # Pass attack specific params
                defense_params={}
            ).to_numpy_client()

            print(f"\n--- 正在测试攻击客户端 {client1_numpy_client.client_id} ---")
            initial_params_att = client1_numpy_client.get_parameters(config={})
            trained_params_att, num_examples_att, metrics_att = client1_numpy_client.fit(initial_params_att, config={})
            print(f"训练完成 (攻击者): {num_examples_att} 个样本，指标: {metrics_att}")
            loss_att, num_eval_examples_att, eval_metrics_att = client1_numpy_client.evaluate(trained_params_att, config={})
            print(f"评估完成 (攻击者): 损失 {loss_att:.4f}，{num_eval_examples_att} 个样本，指标: {eval_metrics_att}")
        else:
            print("客户端 1 数据不足或未创建，跳过攻击客户端测试。")
    else:
        print("未能加载或分配足够的训练数据给客户端0，无法进行客户端单元测试。")