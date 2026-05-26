"""
优化器构建模块

根据 TrainingParams 的配置创建优化器实例.
支持 Adam / SGD / AdamW 三种类型.

各优化器的特点:
- Adam:   自适应学习率,对超参数不敏感,收敛稳定,适合大多数场景
- SGD:    随机梯度下降 + Momentum,调参空间大,泛化能力可能更好
- AdamW:  Adam + 解耦权重衰减,在深度学习任务中通常优于 Adam
"""

from typing import Literal

import torch
import torch.nn as nn

from config.defaults import TrainingParams


def build_optimizer(
    model: nn.Module,
    learning_rate: float = TrainingParams.LEARNING_RATE,
    weight_decay: float = TrainingParams.WEIGHT_DECAY,
    optimizer_type: Literal["Adam", "SGD", "AdamW"] = TrainingParams.OPTIMIZER,
) -> torch.optim.Optimizer:
    """
    构建优化器

    参数说明:
    - learning_rate: 初始学习率.
      太大会导致损失震荡不收敛,太小会收敛过慢.
      Seq2Seq 任务通常从 0.001 或 0.0001 开始尝试.
    - weight_decay: L2 正则化系数.
      对所有权重施加平方惩罚,防止权重过大导致过拟合.
      典型范围 1e-5 ~ 1e-3.
    - optimizer_type: 优化器类型字符串.
      "Adam" / "SGD" / "AdamW"

    Args:
        model:          PyTorch 模型实例(Seq2Seq)
        learning_rate:  初始学习率,默认 0.001(TrainingParams.LEARNING_RATE)
        weight_decay:   权重衰减系数,默认 1e-4(TrainingParams.WEIGHT_DECAY)
        optimizer_type: 优化器类型,默认 "Adam"(TrainingParams.OPTIMIZER)

    Returns:
        配置好的优化器实例

    Raises:
        ValueError: 当 optimizer_type 不在支持列表中时抛出
    """
    if optimizer_type == "Adam":
        # Adam: 结合 Momentum 和 RMSProp,偏差修正使初期更新更稳定
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "SGD":
        # SGD + Momentum: momentum=0.9 能加速收敛并减少震荡
        # 调参时要注意 learning_rate 和 weight_decay 的配合
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    elif optimizer_type == "AdamW":
        # AdamW: 将 weight_decay 从 Adam 的自适应更新中解耦
        # 使正则化效果与学习率调度器的衰减独立,通常泛化更好
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"不支持的优化器类型: '{optimizer_type}',可选: Adam / SGD / AdamW"
        )
