"""
训练工具函数模块

提供随机种子设置、设备选择、参数量统计和时间格式化等通用函数.
所有函数均在模块级别可直接调用,无需实例化.
"""

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from config.defaults import DefaultParams


def set_seed(seed: int = DefaultParams.RANDOM_SEED) -> None:
    """
    设置全局随机种子确保实验可复现

    影响范围:
    - Python random:          影响数据生成的随机性
    - NumPy random:           影响某些预处理中的随机操作
    - PyTorch CPU/CUDA:       影响模型初始化、dropout 等
    - cuDNN deterministic:    消除卷积/RNN 后端中的非确定性实现
    - cuDNN benchmark=False:  关闭自动算法搜索,保证每次运行使用相同实现

    注意:cuDNN deterministic=True 可能会降低性能,但对实验可复现至关重要.
    如果仅需推理可关闭此选项.

    Args:
        seed: 随机种子值,默认 42(DefaultParams.RANDOM_SEED)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 确定性模式:消除后端算法选择的随机性
    torch.backends.cudnn.deterministic = True
    # 关闭 auto-tuner:防止不同运行时选择不同卷积算法
    torch.backends.cudnn.benchmark = False


def get_device(device_string: str = DefaultParams.DEVICE) -> torch.device:
    """
    解析设备字符串并返回 torch.device 对象

    DefaultParams.DEVICE 在模块加载时通过 torch.cuda.is_available() 自动确定:
    GPU 可用时为 "cuda",否则为 "cpu".

    增加二次检查是因为某些边缘场景下模块加载时有 GPU 但运行时没了
    (如容器环境变化),此时自动降级并给出警告.

    Args:
        device_string: "cuda" 或 "cpu",默认自动选择(DefaultParams.DEVICE)

    Returns:
        torch.device 对象
    """
    if device_string == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA 不可用,已自动降级为 CPU")
        return torch.device("cpu")
    return torch.device(device_string)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    统计模型参数数量

    可训练参数 vs 总参数的区别:
    如果模型中某些层被冻结(requires_grad=False),
    可训练参数会少于总参数.
    当前项目不使用参数冻结,两者应相等.

    Args:
        model: PyTorch 模型实例

    Returns:
        (total_params, trainable_params):
            total_params     — 所有参数数量
            trainable_params — 可训练参数数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(elapsed_seconds: float) -> str:
    """
    将秒数格式化为人类可读的时间字符串

    输出格式:
    - >= 3600s:  "Xh Ym Zs"
    - >= 60s:    "Ym Zs"
    - < 60s:     "Zs"

    Args:
        elapsed_seconds: 经过的秒数

    Returns:
        格式化的时间字符串
    """
    minutes, seconds = divmod(int(elapsed_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
