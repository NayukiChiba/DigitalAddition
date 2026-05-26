"""
早停机制模块

监控验证集损失,在连续多轮未显著改善时自动触发停止信号.
避免模型在验证集上不再提升后继续无效训练(过拟合).

与学习率调度器的配合:
EarlyStopping 和 ReduceLROnPlateau 都监控验证损失,
但职责不同:
- ReduceLROnPlateau 发现停滞时先降低学习率("再给模型一次机会")
- EarlyStopping 在学习率已足够低但仍无改善时直接终止训练
建议 EARLY_STOP_PATIENCE > LR_REDUCE_PATIENCE,
即给调度器留出降低学习率的时间窗口.
"""

from typing import Dict, Optional

from config.defaults import TrainingParams


class EarlyStopping:
    """
    早停机制

    判断逻辑:
    - 第一轮:记录基准分数(best_score = val_loss)
    - 后续轮:
      - 如果 val_loss < best_score - min_delta -> 有改善 -> 更新 best_score,重置 counter
      - 如果 val_loss >= best_score - min_delta -> 无改善 -> counter += 1
      - counter >= patience -> 触发停止

    min_delta 的作用:
    防止因浮点精度或微小波动导致的"假改善".
    只有当改善幅度大于 min_delta 时才认为真正在进步.

    Args:
        patience:  容忍轮数,counter 超过此值即触发停止
                   默认 5(TrainingParams.EARLY_STOP_PATIENCE)
        min_delta: 最小改善阈值,小于此值的改善视为未改善
                   默认 1e-4(TrainingParams.EARLY_STOP_MIN_DELTA)

    使用方式:
        early_stop = EarlyStopping()
        for epoch in range(epochs):
            val_loss = train_one_epoch()
            if early_stop(val_loss):
                print("早停触发")
                break
    """

    def __init__(
        self,
        patience: int = TrainingParams.EARLY_STOP_PATIENCE,
        min_delta: float = TrainingParams.EARLY_STOP_MIN_DELTA,
    ):
        self.patience = patience
        self.min_delta = min_delta

        # 连续未改善的轮数
        self.counter: int = 0
        # 历史最佳验证损失(越低越好)
        self.best_score: Optional[float] = None
        # 是否应停止训练
        self.should_stop: bool = False

    def __call__(self, validation_loss: float) -> bool:
        """
        检查并更新早停状态

        Args:
            validation_loss: 当前 epoch 的验证集损失

        Returns:
            True 表示应停止训练,False 表示继续
        """
        # 第一轮:没有可比较的基准,直接记录
        if self.best_score is None:
            self.best_score = validation_loss
        # 当前损失 >= 历史最佳 - delta -> 没有显著改善
        elif validation_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        # 当前损失 < 历史最佳 - delta -> 显著改善,重置 counter
        else:
            self.best_score = validation_loss
            self.counter = 0

        return self.should_stop

    def state_dict(self) -> Dict:
        """
        导出早停状态,随 checkpoint 一起保存

        断点续训时恢复此状态,确保早停逻辑的连续性.
        否则恢复训练后会丢失之前累计的 counter,
        导致本应早停的模型又额外训练 patience 轮.
        """
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        """从 checkpoint 恢复早停状态"""
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.should_stop = state_dict["should_stop"]
