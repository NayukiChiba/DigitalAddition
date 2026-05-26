"""
Checkpoint 管理模块

提供 save_checkpoint 和 load_checkpoint 函数,
用于保存和恢复完整训练状态(模型、优化器、调度器、早停状态、词表等).

设计决策:
- 以字典形式打包保存所有状态为单个 .pth 文件
  而非分开保存多个文件(如 model.pth / optimizer.pth / vocab.json)
  原因:单个文件不会出现"部分文件丢失/版本不一致"的问题,
  恢复训练时只需加载一个文件即可获得完整状态
- 词表保存为 VocabMapping.to_dict() 而非 torch.save 序列化
  原因:to_dict() 产出普通的 Python dict -> JSON 兼容格式,
  torch.save 序列化 torch.Tensor 对词表来说过重且不必要
- load_checkpoint 使用 weights_only=False
  原因:checkpoint 文件包含非张量数据(epoch、validation_loss、vocab dict、
  early_stopping_state dict),PyTorch 2.6+ 默认 weights_only=True
  会拒绝加载这些普通 Python 对象
- eager 模式保存/加载整个 state_dict
  原因:项目规模小(字符级词表 16 个 token),无需考虑分片保存/流式加载

使用方法:
    # 保存
    save_checkpoint(
        filepath=CHECKPOINTS_DIR / "best_model.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=10,
        validation_loss=0.35,
        vocab=vocab,
        early_stopping_state=early_stopping.state_dict(),
    )

    # 加载(恢复训练)
    checkpoint = load_checkpoint(
        filepath=CHECKPOINTS_DIR / "best_model.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda",
    )
    start_epoch = checkpoint["epoch"]
    vocab = VocabMapping.from_dict(checkpoint["vocab"])
"""

from pathlib import Path
from typing import Dict, Optional

import torch

from config.defaults import DefaultParams
from src.data.mapping import VocabMapping


def save_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    validation_loss: float,
    vocab: VocabMapping,
    early_stopping_state: Optional[Dict] = None,
) -> None:
    """
    保存完整训练状态为单个 .pth 文件

    打包内容:
    - epoch: 当前训练轮数(0-based),恢复训练时从此轮继续
    - model_state_dict: 模型权重(Encoder + Decoder 的所有参数)
    - optimizer_state_dict: 优化器状态(Adam 的动量/方差缓存等)
      必须同时保存,否则恢复训练后优化器从零开始累积动量,
      前几个 step 的更新方向会偏离正确轨迹
    - scheduler_state_dict: 学习率调度器状态(StepLR 的步数计数、
      ReduceLROnPlateau 的 patience 计数等)
    - validation_loss: 当前最佳验证损失,用于判断后续 epoch 是否改善
    - vocab: 词表映射(VocabMapping -> dict),恢复时用于 encode/decode
      必须保存词表,因为不同训练运行生成的词表可能不同
    - early_stopping_state(可选): 早停计数器和最佳分数,
      不保存的话恢复训练后早停计数器归零,
      会导致本应早停的模型又被额外训练 patience 轮

    Args:
        filepath: checkpoint 文件保存路径(.pth)
        model: Seq2Seq 模型实例
        optimizer: 优化器实例(含动量缓存等状态)
        scheduler: 学习率调度器实例
        epoch: 当前 epoch 编号(0-based)
        validation_loss: 当前最佳验证集损失
        vocab: 词表映射对象(VocabMapping),内部调用 to_dict() 序列化
        early_stopping_state: 早停状态字典,通过 EarlyStopping.state_dict() 获取
    """
    # 将所有状态打包为单个字典
    # 字典的 key 命名与 load_checkpoint 中一一对应,不要随意改名
    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "validation_loss": validation_loss,
        "vocab": vocab.to_dict(),
    }

    # 早停状态是可选的:首次保存 best_model 时可能还未触发早停
    # 但保存 last_model 时通常已包含早停状态
    if early_stopping_state is not None:
        checkpoint_dict["early_stopping_state"] = early_stopping_state

    # torch.save 使用 pickle 序列化,对 Python dict 和 tensor 混存都适用
    torch.save(checkpoint_dict, filepath)


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: str = DefaultParams.DEVICE,
) -> Dict:
    """
    从 checkpoint 文件恢复训练状态

    恢复策略:
    - model 是必需的:没有模型权重,恢复训练无意义
    - optimizer 和 scheduler 是可选的:
      仅在"恢复训练"时需要加载,若仅用于"推理/评估"则不需要
      因为推理不需要优化器状态
    - 词表不在此函数中恢复为 VocabMapping 对象
      原因:load_checkpoint 返回原始 dict,
      调用方根据自己的需求决定如何重建词表
      (如 VocabMapping.from_dict(checkpoint["vocab"]))

    Args:
        filepath: checkpoint 文件路径(.pth)
        model: 用于加载权重的模型实例,必须与保存时的模型结构一致
        optimizer: 可选,需要恢复状态的优化器实例
        scheduler: 可选,需要恢复状态的调度器实例
        device: 加载设备,"cpu" 或 "cuda"
                即使 checkpoint 在 GPU 上保存,
                map_location 也会自动映射到指定设备

    Returns:
        完整的 checkpoint 字典,包含以下字段:
        - epoch: int
        - model_state_dict: OrderedDict
        - optimizer_state_dict: dict(如保存时包含)
        - scheduler_state_dict: dict(如保存时包含)
        - validation_loss: float
        - vocab: dict(VocabMapping.to_dict() 的产物)
        - early_stopping_state: dict(如保存时包含,可能为 None)

    Raises:
        FileNotFoundError: checkpoint 文件不存在
        KeyError: checkpoint 字典缺少必需字段(文件损坏或版本不兼容)
    """
    # weights_only=False 是必需的:
    # checkpoint 包含 Python dict(vocab、early_stopping_state),
    # PyTorch 默认的 weights_only=True 只允许加载 tensor 类型,
    # 会拒绝这些普通 Python 对象并抛出 UnpicklingError
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # 恢复模型权重:load_state_dict 会严格校验 key 匹配
    # 如果模型结构与保存时不一致(如改了层数/维度),这里会抛出异常
    model.load_state_dict(checkpoint["model_state_dict"])

    # 优化器恢复:需要先加载模型权重再加载优化器状态
    # 因为优化器的 state(如动量缓存)与参数内存地址绑定,
    # 而 load_state_dict(model) 会改变参数的内部存储,
    # 所以必须按"模型 -> 优化器"的顺序恢复
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 调度器恢复:调度器状态独立于模型参数,顺序无特殊要求
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
