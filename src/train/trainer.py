"""
训练器模块

提供 Trainer 类,封装 Seq2Seq 模型的完整训练循环.
包含:训练/验证 epoch、Teacher Forcing、梯度裁剪、学习率调度、
早停检查、checkpoint 自动保存、训练历史记录.

与 EmotionClassification 的 Trainer 区别:
- 处理 Seq2Seq 的 (encoder_input, decoder_input, target_output, encoder_mask) 四元组
- 损失为 CrossEntropyLoss(多分类),而非 BCELoss(二分类)
- 评估指标为完全匹配准确率(Exact Match),而非二分类准确率
"""

import time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.defaults import ModelParams, TrainingParams
from config.paths import BEST_MODEL_PATH, LAST_MODEL_PATH
from src.data.mapping import VocabMapping
from src.train.checkpoint import save_checkpoint
from src.train.early_stopping import EarlyStopping
from src.train.logger import Logger
from src.train.scheduler import is_plateau_scheduler


class Trainer:
    """
    Seq2Seq 训练器

    封装模型训练的完整流程.每个 epoch 执行以下步骤:

    1. train_epoch()      — 训练模式:前向 -> 损失 -> 反向 -> 梯度裁剪 -> 参数更新
    2. validate_epoch()   — 验证模式:计算验证损失 + 完全匹配准确率
    3. scheduler.step()   — 学习率调度(Plateau 类型需传 val_loss)
    4. save_checkpoint()  — 保存最佳/最新模型
    5. early_stopping()   — 早停检查

    所有可配置参数均有来自 TrainingParams 的默认值,
    调用方只需传入 model / data / optimizer / scheduler / criterion / device.

    Args:
        model:                Seq2Seq 模型实例
        train_loader:         训练集 DataLoader
        valid_loader:         验证集 DataLoader
        optimizer:            优化器实例(由 build_optimizer 构建)
        scheduler:            学习率调度器实例(由 build_scheduler 构建)
        criterion:            损失函数,通常是 CrossEntropyLoss(ignore_index=pad_index)
        early_stopping:       早停实例(由 EarlyStopping 构建)
        vocab:                词表映射,随 checkpoint 一同保存以便推理时复用
        device:               计算设备(cuda / cpu)
        epochs:               最大训练轮数,默认来自 TrainingParams.EPOCHS
        grad_clip:            梯度裁剪阈值,默认来自 TrainingParams.GRAD_CLIP
        teacher_forcing_ratio: Teacher Forcing 概率,默认来自 TrainingParams.TEACHER_FORCING_RATIO
    """

    def __init__(
        self,
        # --- 必需参数(无默认值) ---
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        criterion: nn.Module,
        early_stopping: EarlyStopping,
        vocab: VocabMapping,
        device: torch.device,
        # --- 可选参数(均来自 TrainingParams) ---
        epochs: int = TrainingParams.EPOCHS,
        grad_clip: float = TrainingParams.GRAD_CLIP,
        teacher_forcing_ratio: float = ModelParams.TEACHER_FORCING_RATIO,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.early_stopping = early_stopping
        self.vocab = vocab
        self.device = device
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # 日志记录器:同时写入 CSV 和 TensorBoard(若安装了 tensorboard)
        self.logger = Logger()

        # 训练历史:逐 epoch 记录,train() 结束后返回
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        # 追踪最佳验证损失,用于决定何时保存最佳模型
        self.best_val_loss = float("inf")

    # ==================================================================
    # 训练 epoch
    # ==================================================================
    def train_epoch(self, epoch: int) -> float:
        """
        执行一个训练 epoch

        数据处理流程(每个 batch):
        encoder_input:  [batch_size, encoder_length]     ← 如 [7, 9, 14, 6, 11]
        decoder_input:  [batch_size, decoder_length]     ← 如 [2, 10, 6]
        target_output:  [batch_size, decoder_length]     ← 如 [10, 6, 3]
        encoder_mask:   [batch_size, encoder_length]     ← True=有效, False=PAD
            │
            ├─-> model.forward(teacher_forcing_ratio)
            │   logits: [batch_size, decoder_length, vocab_size]
            │
            ├─-> reshape + CrossEntropyLoss(ignore_index=0)
            │   logits_flat: [batch_size × decoder_length, vocab_size]
            │   target_flat: [batch_size × decoder_length]
            │
            └─-> loss.backward() -> grad_clip -> optimizer.step()

        梯度裁剪原因:
        RNN 训练中梯度容易爆炸(vanishing/exploding gradients),
        通过 clip_grad_norm_ 将梯度的 L2 范数限制在 grad_clip 以内.

        Teacher Forcing 原理:
        以概率 teacher_forcing_ratio 使用真实的 target token 作为 decoder 下一步输入,
        否则用模型自己的预测.这能加速收敛,同时让模型学会纠正自身错误.

        Args:
            epoch: 当前 epoch 编号(0-based,进度条显示 +1)

        Returns:
            该 epoch 的平均训练损失(标量)
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        desc = f"[Train] Epoch {epoch + 1}/{self.epochs}"
        progress_bar = tqdm(self.train_loader, desc=desc, unit="batch")

        for encoder_input, decoder_input, target_output, encoder_mask in progress_bar:
            # 将数据从 CPU 移到 GPU(或保持 CPU)
            encoder_input = encoder_input.to(self.device)
            decoder_input = decoder_input.to(self.device)
            target_output = target_output.to(self.device)
            encoder_mask = encoder_mask.to(self.device)

            batch_size = encoder_input.size(0)

            # --- 前向传播 ---
            # 清空上一轮的梯度缓存
            self.optimizer.zero_grad()

            # Seq2Seq 前向:Encoder 编码 + Teacher Forcing 驱动 Decoder
            # 输出形状: [batch_size, decoder_length, vocab_size]
            # 每个位置给出词表中每个 token 的 logit 分数
            logits = self.model(
                encoder_input,
                decoder_input,
                encoder_mask,
                self.teacher_forcing_ratio,
            )

            # --- 损失计算 ---
            # CrossEntropyLoss 需要 (N, num_classes) 的预测和 (N,) 的目标
            # 将 batch + 时间步展平为一维
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_output.reshape(-1)

            # ignore_index=pad_index 确保 PAD 位置不贡献梯度
            # 因为 collate_fn 中 PAD 位置填了 0,而 pad_index=0
            loss = self.criterion(logits_flat, target_flat)

            # --- 反向传播 ---
            loss.backward()

            # 梯度裁剪:限制所有参数的梯度 L2 范数上限
            # 防止 RNN 中的梯度爆炸导致训练不稳定
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # 参数更新
            self.optimizer.step()

            # --- 累计统计 ---
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
            )

        return total_loss / total_samples

    # ==================================================================
    # 验证 epoch
    # ==================================================================
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> tuple:
        """
        执行一个验证 epoch

        每个 batch 做两件事:
        1. 计算验证损失(Teacher Forcing ratio=1.0,全用真实 token)
        2. 计算完全匹配准确率(贪心解码,逐样本比对)

        为什么验证损失用 Teacher Forcing 而准确率用贪心解码？
        - Teacher Forcing 损失更稳定,每次评估结果一致,适合做早停和模型选择
        - 贪心解码准确率反映模型在真实推理场景下的表现,更直观

        完全匹配准确率:
        贪心解码生成完整序列 generated_ids,
        与 target_output 逐样本逐位置比较,
        整条序列全部一致才算正确(Exact Match).

        Args:
            epoch: 当前 epoch 编号(0-based)

        Returns:
            (val_loss, val_accuracy):
                val_loss     — 平均验证损失
                val_accuracy — 完全匹配准确率,范围 [0, 1]
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        desc = f"[Valid] Epoch {epoch + 1}/{self.epochs}"
        progress_bar = tqdm(self.valid_loader, desc=desc, unit="batch")

        for encoder_input, decoder_input, target_output, encoder_mask in progress_bar:
            encoder_input = encoder_input.to(self.device)
            decoder_input = decoder_input.to(self.device)
            target_output = target_output.to(self.device)
            encoder_mask = encoder_mask.to(self.device)

            batch_size = encoder_input.size(0)

            # ==========================================================
            # 验证损失:Teacher Forcing 模式(ratio=1.0)
            # 每次 decoder 输入都是真实的 target token,
            # 不引入模型自身预测的误差,损失值稳定可复现
            # ==========================================================
            logits = self.model(
                encoder_input,
                decoder_input,
                encoder_mask,
                teacher_forcing_ratio=1.0,
            )
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = target_output.reshape(-1)
            loss = self.criterion(logits_flat, target_flat)
            total_loss += loss.item() * batch_size

            # ==========================================================
            # 完全匹配准确率:贪心解码模式
            # 用 Seq2Seq.generate() 自回归生成完整序列,
            # 与 target_output 逐样本逐位置比对
            # ==========================================================
            generated_ids, _ = self.model.generate(
                encoder_input,
                encoder_mask,
                max_generation_length=target_output.size(1),
            )

            # 逐样本比较
            for i in range(batch_size):
                target_seq = target_output[i]  # 真实序列
                pred_seq = generated_ids[i, : target_seq.size(0)]  # 截取等长部分
                if torch.equal(pred_seq, target_seq):
                    total_correct += 1

            total_samples += batch_size

            progress_bar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                acc=f"{total_correct / total_samples:.4f}",
            )

        return (
            total_loss / total_samples,
            total_correct / total_samples,
        )

    # ==================================================================
    # 完整训练流程
    # ==================================================================
    def train(self) -> Dict[str, List[float]]:
        """
        执行完整训练流程

        每个 epoch 依次执行:
        1. train_epoch()      — 训练一轮
        2. validate_epoch()   — 验证一轮(损失 + 准确率)
        3. 记录历史            — 追加到 self.history
        4. scheduler.step()   — 调整学习率
        5. 控制台输出          — 打印 epoch 摘要
        6. logger.log_epoch() — 写入 CSV + TensorBoard
        7. save_checkpoint()  — 保存最佳/最新模型
        8. early_stopping()   — 检查是否该停了

        最佳模型 vs 最新模型的区别:
        - best_model.pth: 验证损失最低的那个 epoch 的权重,用于最终评估
        - last_model.pth: 最后一个 epoch 的权重,用于断点续训

        Returns:
            history: 训练历史字典
                     {"train_loss": [...], "val_loss": [...], "val_accuracy": [...]}
        """
        # --- 训练开始 ---
        print(f"开始训练: epochs={self.epochs}, device={self.device}")
        print(f"Teacher Forcing ratio: {self.teacher_forcing_ratio}")

        self.logger.start()
        start_time = time.time()

        for epoch in range(self.epochs):
            # --- Step 1: 训练 ---
            train_loss = self.train_epoch(epoch)

            # --- Step 2: 验证 ---
            val_loss, val_accuracy = self.validate_epoch(epoch)

            # --- Step 3: 记录历史 ---
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            # --- Step 4: 学习率调度 ---
            # 获取当前学习率(调度器可能已调整)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ReduceLROnPlateau 需要传入验证损失以判断是否衰减
            # StepLR / CosineAnnealingLR 无需参数
            if is_plateau_scheduler(self.scheduler):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # --- Step 5: 控制台输出 ---
            print(
                f"Epoch {epoch + 1:3d}/{self.epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_accuracy:.4f} | "
                f"lr: {current_lr:.2e}"
            )

            # --- Step 6: CSV + TensorBoard 日志 ---
            self.logger.log_epoch(
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": current_lr,
                },
                epoch=epoch,
            )

            # --- Step 7: Checkpoint 保存 ---

            # 最佳模型:仅当验证损失创新低时覆盖保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    BEST_MODEL_PATH,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_loss,
                    self.vocab,
                    self.early_stopping.state_dict(),
                )
                print(f"  >> 最佳模型已保存 (val_loss={val_loss:.4f})")

            # 最新模型:每个 epoch 都保存,便于意外中断后从断点恢复
            save_checkpoint(
                LAST_MODEL_PATH,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                val_loss,
                self.vocab,
                self.early_stopping.state_dict(),
            )

            # --- Step 8: 早停检查 ---
            if self.early_stopping(val_loss):
                print(
                    f"早停触发: 验证损失在 {self.early_stopping.patience} 轮内未显著改善"
                )
                break

        # --- 训练结束 ---
        elapsed = time.time() - start_time
        minutes, seconds = divmod(int(elapsed), 60)
        print(f"训练完成,总耗时: {minutes}m {seconds}s")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")

        # 关闭日志
        self.logger.log_message(
            f"训练结束,epoch={epoch + 1}, "
            f"best_val_loss={self.best_val_loss:.4f}, "
            f"time={minutes}m{seconds}s"
        )
        self.logger.close()

        return self.history
