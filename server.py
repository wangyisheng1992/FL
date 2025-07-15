from flwr.server.strategy import Strategy
from flwr.common import Parameters, FitRes, EvaluateRes
from typing import List, Tuple, Dict, Optional
import torch
import flwr as fl
# 假设 aggregate_weights_krum 和 aggregate_weights_multi_krum 在 defense.py 中定义
from defense import aggregate_weights_krum, aggregate_weights_multi_krum 

class CustomStrategy(fl.server.strategy.FedAvg):
    """
    一个自定义的 Flower 策略，扩展了 FedAvg，并支持 Krum 或 Multi-Krum 聚合防御机制。
    根据初始化的 `defense_type` 参数，它将在 `aggregate_fit` 阶段应用不同的聚合逻辑。
    """
    def __init__(self, defense_type: str, *args, **kwargs):
        """
        初始化自定义策略。
        
        Args:
            defense_type (str): 指定要使用的防御类型。
                                 目前支持 "krum"、"multi_krum" 或其他任何字符串（将回退到 FedAvg）。
            *args, **kwargs: 传递给基类 fl.server.strategy.FedAvg 的参数。
        """
        super().__init__(*args, **kwargs)
        self.defense_type = defense_type
        print(f"自定义策略已初始化，聚合防御类型为: {self.defense_type}")

    def aggregate_fit(
        self,
        rnd: int, # 当前的联邦学习轮次
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]], # 从客户端收到的训练结果列表
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]], # 训练失败的客户端列表
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        聚合客户端在训练轮次中返回的模型更新。
        根据 `defense_type` 选择 Krum、Multi-Krum 或默认的 FedAvg 聚合方式。
        """
        # 如果有失败的客户端，通常会打印警告或处理
        if failures:
            print(f"聚合适应阶段：服务器轮次 {rnd} 中出现失败: {failures}")

        # 如果没有收到任何结果，则无法聚合
        if not results:
            print(f"聚合适应阶段：服务器轮次 {rnd} 未收到任何训练结果，跳过聚合。")
            return None, {}

        # 根据 defense_type 应用不同的聚合逻辑
        if self.defense_type == "krum":
            print(f"服务器轮次 {rnd}: 正在使用 Krum 聚合。")
            # aggregate_weights_krum 需要一个字典，键是客户端ID，值是模型的参数张量列表
            # FitRes.parameters 是一个 Parameters 对象，其 tensors 属性是 List[bytes]
            # 您的 defense.py 中的 aggregate_weights_krum 可能需要将这些 bytes 转换回 PyTorch 张量或 NumPy 数组
            return aggregate_weights_krum({client.cid: res.parameters.tensors for client, res in results})
        elif self.defense_type == "multi_krum":
            print(f"服务器轮次 {rnd}: 正在使用 Multi-Krum 聚合。")
            # 同样，aggregate_weights_multi_krum 也需要处理参数的格式
            return aggregate_weights_multi_krum({client.cid: res.parameters.tensors for client, res in results})
        else:
            print(f"服务器轮次 {rnd}: 正在使用默认的 FedAvg 聚合 (defense_type='{self.defense_type}')。")
            # 调用基类 FedAvg 的 aggregate_fit 方法
            return super().aggregate_fit(rnd, results, failures)

    # configure_fit, num_fit_clients, num_evaluation_clients, configure_evaluate, aggregate_evaluate, evaluate
    # 等方法可以直接继承自 fl.server.strategy.FedAvg，或者根据需要进行覆盖。
    # 这里我们只重写了 aggregate_fit。