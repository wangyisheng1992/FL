# strategy.py
import flwr as fl
from flwr.server.strategy.aggregate import aggregate
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays, Status, Code
from typing import List, Tuple, Dict, Optional, Union, Callable
import numpy as np

class Krum(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        num_malicious_clients: int = 1,
        num_clients_to_aggregate: int = 1,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[Callable[[int, fl.common.NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        adaptive_threshold: bool = True,
        distance_metric: str = "cosine",
    ):
        super().__init__()
        # 参数有效性检查
        if not (0.0 <= fraction_fit <= 1.0):
            raise ValueError(f"fraction_fit_clients 必须在 0.0 到 1.0 之间 (得到: {fraction_fit})")
        if not (0.0 <= fraction_evaluate <= 1.0):
            raise ValueError(f"fraction_evaluate_clients 必须在 0.0 到 1.0 之间 (得到: {fraction_evaluate})")

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_aggregate = num_clients_to_aggregate
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.adaptive_threshold = adaptive_threshold
        self.distance_metric = distance_metric
        self.history_scores = []

    def __repr__(self) -> str:
        # 对象的字符串表示形式
        return f"Krum(恶意客户端数量={self.num_malicious_clients}, 聚合客户端数量={self.num_clients_to_aggregate})"

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """初始化全局模型参数。"""
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """配置下一轮训练任务。"""
        # 计算需要抽样的客户端数量
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        # 从可用客户端中抽样
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        # 创建训练指令 (FitIns)，包含当前全局参数和空配置
        fit_ins = fl.common.FitIns(parameters, {}) 
        # 返回客户端及其对应的训练指令列表
        return [(client, fit_ins) for client in clients]

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """返回要用于训练的客户端数量 (抽样大小和最小客户端数)。"""
        num_clients = int(num_available_clients * self.fraction_fit)
        # 确保抽样数量不低于最小训练客户端数，且不低于最小可用客户端数
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """返回要用于评估的客户端数量 (抽样大小和最小客户端数)。"""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        # 确保抽样数量不低于最小评估客户端数，且不低于最小可用客户端数
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
        
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """配置下一轮评估任务。"""
        # 如果评估比例为 0，则不进行评估
        if self.fraction_evaluate == 0.0:
            return []
        
        # 计算需要抽样的客户端数量
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        # 从可用客户端中抽样
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        # 创建评估指令 (EvaluateIns)，包含当前全局参数和空配置
        eval_ins = fl.common.EvaluateIns(parameters, {}) 
        # 返回客户端及其对应的评估指令列表
        return [(client, eval_ins) for client in clients]

    def _compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量之间的距离。"""
        if self.distance_metric == "cosine":
            # 余弦距离
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return 1 - np.dot(vec1, vec2) / (norm1 * norm2)
        else:
            # 默认使用欧氏距离
            return np.sum(np.square(vec1 - vec2))

    def _detect_outliers(self, distances: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """检测异常值。"""
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        if std_dist == 0:
            return np.zeros_like(distances, dtype=bool)
        z_scores = np.abs((distances - mean_dist) / std_dist)
        return z_scores > threshold

    def _compute_adaptive_threshold(self, scores: List[float]) -> float:
        """计算自适应阈值。"""
        if not self.history_scores:
            return np.mean(scores) + 2 * np.std(scores)
        
        # 使用历史分数和当前分数的组合
        all_scores = self.history_scores + scores
        return np.mean(all_scores) + 2 * np.std(all_scores)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """使用增强的 Krum/Multi-Krum 聚合模型更新。"""
        if not results:
            print(f"服务器轮次 {server_round}: 未收到任何训练结果，跳过聚合。")
            return None, {}
        if failures:
            print(f"服务器轮次 {server_round}: 训练过程中出现失败: {failures}")

        weights_results = []
        for _, fit_res in results:
            weights_results.append(
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            )
        
        num_received_updates = len(weights_results)
        if num_received_updates < self.min_fit_clients:
            print(f"Krum: 收到的更新不足 ({num_received_updates} < {self.min_fit_clients})。跳过聚合。")
            return None, {}

        # 将所有客户端的模型更新参数展平为一维向量
        vec_updates = [np.concatenate([arr.flatten() for arr in params]) for params, _ in weights_results]
        num_selected_clients = len(vec_updates)

        if num_selected_clients == 0:
            return None, {}

        # 计算距离矩阵
        distances = np.zeros((num_selected_clients, num_selected_clients))
        for i in range(num_selected_clients):
            for j in range(i + 1, num_selected_clients):
                dist = self._compute_distance(vec_updates[i], vec_updates[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # 检测异常值
        outlier_mask = self._detect_outliers(distances.flatten())
        if np.any(outlier_mask):
            print(f"检测到 {np.sum(outlier_mask)} 个异常值")

        k_neighbors = max(0, num_selected_clients - self.num_malicious_clients - 2)
        
        if k_neighbors < 0:
            print(f"警告: Krum 的 k_neighbors 为 {k_neighbors}。至少需要 {self.num_malicious_clients + 2} 个客户端才能进行筛选。")
            if not weights_results: return None, {}
            aggregated_ndarrays = aggregate([(params, num_ex) for params, num_ex in weights_results])
            return ndarrays_to_parameters(aggregated_ndarrays), {"krum_fallback_fedavg": 1}

        # 计算 Krum 分数
        scores = []
        for i in range(num_selected_clients):
            client_distances = np.sort(distances[i, :])
            if num_selected_clients == 1:
                scores.append(0)
            else:
                actual_k_neighbors = min(k_neighbors, len(client_distances) - 1)
                if actual_k_neighbors > 0:
                    scores.append(np.sum(client_distances[1 : 1 + actual_k_neighbors]))
                else:
                    scores.append(0)

        # 使用自适应阈值
        if self.adaptive_threshold:
            threshold = self._compute_adaptive_threshold(scores)
            valid_indices = [i for i, score in enumerate(scores) if score <= threshold]
            if not valid_indices:
                print("警告: 自适应阈值过滤后没有有效的客户端更新")
                return None, {}
            selected_indices = valid_indices[:self.num_clients_to_aggregate]
        else:
            sorted_indices = np.argsort(scores)
            selected_indices = sorted_indices[:self.num_clients_to_aggregate]

        # 更新历史分数
        self.history_scores = scores[-10:]  # 只保留最近10轮的分数

        # 聚合选定的更新
        selected_updates = [weights_results[i][0] for i in selected_indices]
        if not selected_updates:
            print("Krum: 没有选中任何更新进行聚合。")
            return None, {}

        # 加权平均聚合
        weights = [1.0 / (scores[i] + 1e-10) for i in selected_indices]  # 使用分数的倒数作为权重
        weights = np.array(weights) / np.sum(weights)  # 归一化权重

        aggregated_ndarrays = [
            np.sum([update[layer_idx] * w for update, w in zip(selected_updates, weights)], axis=0)
            for layer_idx in range(len(selected_updates[0]))
        ]
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
        metrics_aggregated = {
            "selected_clients": selected_indices,
            "krum_scores": [scores[i] for i in selected_indices],
            "adaptive_threshold": threshold if self.adaptive_threshold else None
        }
        
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """聚合评估结果 (标准的加权平均损失和准确率)。"""
        if not results:
            print(f"服务器轮次 {server_round}: 未收到任何评估结果，跳过聚合评估。")
            return None, {}
        if failures:
            print(f"服务器轮次 {server_round}: 评估过程中出现失败: {failures}")

        # 聚合损失 (加权平均)
        loss_aggregated = fl.server.strategy.aggregate.weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )
        # 聚合准确率 (加权平均，这里假设准确率作为 metrics['accuracy'])
        accuracy_aggregated = fl.server.strategy.aggregate.weighted_loss_avg( 
            [(evaluate_res.num_examples, evaluate_res.metrics["accuracy"]) for _, evaluate_res in results]
        )
        metrics_aggregated = {"accuracy": accuracy_aggregated}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """在服务器端评估模型 (如果需要，例如在中央测试集上)。"""
        if self.evaluate_fn is None:
            # 没有提供评估函数
            return None
        
        # 将 Parameters 转换为 NDArrays
        ndarrays = parameters_to_ndarrays(parameters)
        
        # 调用提供的评估函数
        # 注意：config 参数这里传递了一个空字典，与 FedAvg 行为一致
        eval_res = self.evaluate_fn(server_round, ndarrays, {})
        
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

class CoordMedian(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[Callable[[int, fl.common.NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]] = None,
        use_weights: bool = True,
        adaptive_lr: bool = True,
    ):
        super().__init__()
        if not (0.0 <= fraction_fit <= 1.0):
            raise ValueError(f"fraction_fit must be between 0.0 and 1.0 (got: {fraction_fit})")
        if not (0.0 <= fraction_evaluate <= 1.0):
            raise ValueError(f"fraction_evaluate must be between 0.0 and 1.0 (got: {fraction_evaluate})")

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.use_weights = use_weights
        self.adaptive_lr = adaptive_lr
        self.history_losses = []

    def __repr__(self) -> str:
        return f"CoordMedian()"

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure the next round of training."""
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        
        fit_ins = fl.common.FitIns(parameters, {})
        return [(client, fit_ins) for client in clients]

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
        
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        eval_ins = fl.common.EvaluateIns(parameters, {})
        return [(client, eval_ins) for client in clients]

    def _compute_client_weights(self, updates: List[np.ndarray], num_examples: List[int]) -> np.ndarray:
        """计算客户端权重。"""
        if not self.use_weights:
            return np.ones(len(updates)) / len(updates)
        
        # 计算每个更新的范数
        norms = [np.linalg.norm(np.concatenate([arr.flatten() for arr in update])) for update in updates]
        # 使用范数的倒数作为权重
        weights = 1.0 / (np.array(norms) + 1e-10)
        # 结合样本数量
        weights = weights * np.array(num_examples)
        return weights / np.sum(weights)

    def _compute_adaptive_lr(self, current_loss: float) -> float:
        """计算自适应学习率。"""
        if not self.adaptive_lr or not self.history_losses:
            return 1.0
        
        # 如果损失增加，降低学习率
        if current_loss > np.mean(self.history_losses):
            return 0.5
        # 如果损失减少，增加学习率
        return 1.5

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """使用增强的坐标中位数聚合模型更新。"""
        if not results:
            print(f"CoordMedian (Round {server_round}): No results received, skipping aggregation.")
            return None, {}
        if failures:
            print(f"CoordMedian (Round {server_round}): Failures occurred: {failures}")

        # 获取更新和样本数量
        updates = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_examples = [fit_res.num_examples for _, fit_res in results]
        
        if not updates:
            print(f"CoordMedian (Round {server_round}): No updates to aggregate after processing results.")
            return None, {}

        # 计算客户端权重
        weights = self._compute_client_weights(updates, num_examples)
        
        # 计算当前轮次的平均损失
        current_loss = np.mean([fit_res.metrics.get("loss", 0.0) for _, fit_res in results])
        self.history_losses.append(current_loss)
        if len(self.history_losses) > 10:  # 只保留最近10轮的损失
            self.history_losses.pop(0)
        
        # 计算自适应学习率
        lr = self._compute_adaptive_lr(current_loss)

        # 聚合每一层的参数
        num_layers = len(updates[0])
        aggregated_ndarrays: List[np.ndarray] = []

        for i in range(num_layers):
            # 堆叠当前层的所有更新
            layer_updates_stacked = np.stack([update[i] for update in updates])
            
            # 计算加权中位数
            if self.use_weights:
                # 对每个参数位置分别计算加权中位数
                median_for_layer = np.zeros_like(layer_updates_stacked[0])
                for idx in np.ndindex(layer_updates_stacked[0].shape):
                    values = layer_updates_stacked[(slice(None),) + idx]
                    median_for_layer[idx] = np.average(values, weights=weights)
            else:
                # 使用普通中位数
                median_for_layer = np.median(layer_updates_stacked, axis=0)
            
            # 应用自适应学习率
            median_for_layer = median_for_layer * lr
            
            aggregated_ndarrays.append(median_for_layer.astype(updates[0][i].dtype))

        if not aggregated_ndarrays:
            print(f"CoordMedian (Round {server_round}): Aggregated ndarrays are empty.")
            return None, {}
             
        print(f"CoordMedian (Round {server_round}): Successfully aggregated {len(updates)} client updates.")
        metrics = {
            "adaptive_lr": lr,
            "current_loss": current_loss,
            "mean_loss": np.mean(self.history_losses) if self.history_losses else None
        }
        return ndarrays_to_parameters(aggregated_ndarrays), metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results (standard weighted average)."""
        if not results:
            print(f"CoordMedian (Round {server_round}): No evaluation results received.")
            return None, {}
        if failures:
            print(f"CoordMedian (Round {server_round}): Evaluation failures: {failures}")

        loss_aggregated = fl.server.strategy.aggregate.weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )
        
        # Aggregate custom metrics if present, e.g., accuracy
        metrics_aggregated = {}
        # Check if 'accuracy' is present in all results to avoid errors
        if all("accuracy" in evaluate_res.metrics for _, evaluate_res in results):
            accuracy_aggregated = fl.server.strategy.aggregate.weighted_loss_avg( 
                [(evaluate_res.num_examples, evaluate_res.metrics["accuracy"]) for _, evaluate_res in results if "accuracy" in evaluate_res.metrics]
            )
            metrics_aggregated["accuracy"] = accuracy_aggregated
        else:
            print(f"CoordMedian (Round {server_round}): 'accuracy' metric not found in all evaluation results. Skipping accuracy aggregation.")
            
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model on the server side (if evaluate_fn is provided)."""
        if self.evaluate_fn is None:
            return None
        
        ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, ndarrays, {}) # Pass an empty config dict
        
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

if __name__ == '__main__':
    print("--- 简单测试 Krum 聚合逻辑 (不启动 Flower 服务) ---")

    # 创建一些虚拟的模型参数 (NumPy NdArrays)
    def create_dummy_params(seed=None):
        if seed is not None: np.random.seed(seed)
        # 创建两个随机数组作为参数，模拟模型的两层
        return [np.random.rand(2,2).astype(np.float32), np.random.rand(3).astype(np.float32)]

    # 创建一组模拟的客户端模型参数
    # params1, params2, params4 模拟"好"客户端的更新
    # params3, params5 模拟"恶意"客户端的更新 (通过不同的随机种子使其更远)
    params1 = create_dummy_params(0) # "好" 客户端 1
    params2 = create_dummy_params(1) # "好" 客户端 2
    params3 = create_dummy_params(10) # "恶意" 客户端 1 (更新偏离较大)
    params4 = create_dummy_params(2) # "好" 客户端 3
    params5 = create_dummy_params(11) # "恶意" 客户端 2 (更新偏离较大)

    # 模拟从客户端收到的 FitRes 结果
    # FitRes 包含客户端的参数和训练样本数量
    results_fit = [
        (None, FitRes(parameters=ndarrays_to_parameters(params1), num_examples=10, metrics={}, status=Status(code=Code.OK, message="成功"))),
        (None, FitRes(parameters=ndarrays_to_parameters(params2), num_examples=10, metrics={}, status=Status(code=Code.OK, message="成功"))),
        (None, FitRes(parameters=ndarrays_to_parameters(params3), num_examples=10, metrics={}, status=Status(code=Code.OK, message="成功"))),
        (None, FitRes(parameters=ndarrays_to_parameters(params4), num_examples=10, metrics={}, status=Status(code=Code.OK, message="成功"))),
        (None, FitRes(parameters=ndarrays_to_parameters(params5), num_examples=10, metrics={}, status=Status(code=Code.OK, message="成功"))),
    ]
    
    # --- 测试 Krum (选择 1 个客户端进行聚合) ---
    # 假设有 1 个恶意客户端 (f=1)。Krum 理论上要求 n >= 2f + 3，这里 n=5，2*1+3=5。刚好满足条件。
    # 根据 Krum 原始论文，Krum 分数计算需要求和到 n-f-1 个最近邻居的距离。
    # 在这个实现中，k_neighbors = num_selected_clients - num_malicious_clients - 2。
    # 对于 5 个客户端，f=1，则 k_neighbors = 5 - 1 - 2 = 2。
    print("\n--- Krum 策略测试 (选择 1 个客户端聚合) ---")
    krum_strategy_single = Krum(min_fit_clients=2, num_malicious_clients=1, num_clients_to_aggregate=1)
    agg_params_single, _ = krum_strategy_single.aggregate_fit(server_round=1, results=results_fit, failures=[])
    
    if agg_params_single:
        print("Krum (选择 1) 聚合后的参数 (第一层参数示例):")
        print(parameters_to_ndarrays(agg_params_single)[0])
    else:
        print("Krum (选择 1) 未能聚合参数。")

    # --- 测试 Multi-Krum (选择 2 个客户端进行聚合) ---
    print("\n--- Multi-Krum 策略测试 (选择 2 个客户端聚合) ---")
    krum_strategy_multi = Krum(min_fit_clients=2, num_malicious_clients=1, num_clients_to_aggregate=2)
    agg_params_multi, _ = krum_strategy_multi.aggregate_fit(server_round=1, results=results_fit, failures=[])

    if agg_params_multi:
        print("Multi-Krum (选择 2) 聚合后的参数 (第一层参数示例):")
        print(parameters_to_ndarrays(agg_params_multi)[0])
    else:
        print("Multi-Krum (选择 2) 未能聚合参数。")

    print("\n--- 期望结果 ---")
    print("被 Krum/Multi-Krum 选中的参数应该更接近 params1、params2、params4 (即 '好' 客户端的更新)。")
    print("params3 和 params5 (即 '恶意' 客户端的更新) 应该因为与其他更新距离较远而被 Krum 算法识别并赋予较高的 Krum 分数，从而被排除或权重降低。")