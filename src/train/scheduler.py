"""
学习率调度器模块

根据 TrainingParams 的配置创建学习率调度器实例.
支持 StepLR / CosineAnnealingLR / ReduceLROnPlateau 三种策略.

调度策略选择建议:
- StepLR:              简单可控,每 N 轮衰减固定比例,适合快速实验
- CosineAnnealingLR:   平滑衰减到接近 0,在图像/大模型任务中常见
- ReduceLROnPlateau:   自适应,验证损失不再下降时自动衰减 lr,推荐作为默认

提供 is_plateau_scheduler() 辅助函数,
用于 Trainer 中判断是否需要传 val_loss 给 scheduler.step().
"""

from typing import Literal

import torch

from config.defaults import TrainingParams


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Literal[
        "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"
    ] = TrainingParams.LR_SCHEDULER,
    step_size: int = TrainingParams.LR_STEP_SIZE,
    gamma: float = TrainingParams.LR_GAMMA,
    reduce_factor: float = TrainingParams.LR_REDUCE_FACTOR,
    reduce_patience: int = TrainingParams.LR_REDUCE_PATIENCE,
    epochs: int = TrainingParams.EPOCHS,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    构建学习率调度器

    Args:
        optimizer:       已构建的优化器实例
        scheduler_type:  "StepLR" / "CosineAnnealingLR" / "ReduceLROnPlateau"
                         默认 "StepLR"(TrainingParams.LR_SCHEDULER)
        step_size:       StepLR 衰减周期(每隔 step_size 个 epoch 衰减一次)
                         默认 10(TrainingParams.LR_STEP_SIZE)
        gamma:           StepLR 衰减因子,新 lr = 旧 lr × gamma
                         默认 0.5(TrainingParams.LR_GAMMA)
        reduce_factor:   ReduceLROnPlateau 衰减因子
                         默认 0.5(TrainingParams.LR_REDUCE_FACTOR)
        reduce_patience: ReduceLROnPlateau 容忍轮数
                         默认 3(TrainingParams.LR_REDUCE_PATIENCE)
        epochs:          总训练轮数,CosineAnnealingLR 的完整余弦周期
                         默认 50(TrainingParams.EPOCHS)

    Returns:
        配置好的调度器实例

    Raises:
        ValueError: 当 scheduler_type 不在支持列表中时抛出

    注意:
        ReduceLROnPlateau 的 step() 需要传入验证损失:
            scheduler.step(val_loss)
        StepLR / CosineAnnealingLR 只需:
            scheduler.step()
    """
    if scheduler_type == "StepLR":
        # StepLR: 每隔固定 epoch 学习率乘以 gamma
        # 如 step_size=10, gamma=0.5: lr 每 10 轮对半衰减
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_type == "CosineAnnealingLR":
        # CosineAnnealingLR: 学习率沿余弦曲线从初始值平滑衰减到 eta_min
        # T_max=epochs 使一个完整周期刚好覆盖整个训练过程
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=0,  # 最终衰减到 0
        )
    elif scheduler_type == "ReduceLROnPlateau":
        # ReduceLROnPlateau: 监控验证损失,连续 reduce_patience 轮不改善则衰减
        # mode="min": 指标越低越好
        # min_lr=1e-6: 学习率不会无限衰减,设一个合理的下限
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=reduce_factor,
            patience=reduce_patience,
            min_lr=1e-6,
        )
    else:
        raise ValueError(
            f"不支持的调度器类型: '{scheduler_type}',"
            f"可选: StepLR / CosineAnnealingLR / ReduceLROnPlateau"
        )


def is_plateau_scheduler(scheduler: torch.optim.lr_scheduler.LRScheduler) -> bool:
    """
    判断调度器是否为 ReduceLROnPlateau 类型

    ReduceLROnPlateau 的 step() 签名与其他调度器不同,
    需要传入验证损失值作为判断依据.
    Trainer 中根据此函数的返回值决定调用 scheduler.step() 还是 scheduler.step(val_loss).

    Args:
        scheduler: 调度器实例

    Returns:
        True 表示需要传 val_loss
    """
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
