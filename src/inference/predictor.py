"""
推理模块

提供 Predictor 类，封装训练好的 Seq2Seq 模型对单条/批量/文件输入
进行推理预测的完整流程。

与训练时的区别：
- 不使用 Teacher Forcing，全程自回归贪心解码
- 关闭梯度计算 (@torch.no_grad)，节省显存并加速
- 输入为原始字符串（如 "35+27"），输出为字符串（如 "62"），
  内部自动完成 encode -> generate -> decode 的转换

设计决策：
- from_checkpoint 类方法独立加载模型 + 词表，不依赖 trainer 模块
  原因：推理环境可能不安装训练依赖（如 tensorboard），
  且推理只需要模型权重和词表，不需要优化器/调度器等训练状态
- 单条预测与批量预测分离：
  predict() 处理单条字符串，返回单个字符串
  predict_batch() 处理字符串列表，返回字符串列表
  两者共用内部 _predict_tensor() 方法处理张量运算
"""

import csv
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

from config.defaults import DefaultParams, InferenceParams
from src.data.mapping import VocabMapping
from src.model import build_model


class Predictor:
    """
    Seq2Seq 算术推理器

    加载训练好的模型和词表，将输入表达式转换为计算结果。

    使用方式:
        # 方式1: 从组件构建
        predictor = Predictor(model, vocab, device)

        # 方式2: 从 checkpoint 恢复
        predictor = Predictor.from_checkpoint("outputs/checkpoints/best_model.pth")

        # 单条预测
        result = predictor.predict("35+27")  # -> "62"

        # 批量预测
        results = predictor.predict_batch(["35+27", "80-12"])  # -> ["62", "68"]

        # 文件预测
        predictor.predict_file("input.csv", "output.csv", "expression")
    """

    def __init__(
        self,
        model: nn.Module,
        vocab: VocabMapping,
        device: torch.device | str = DefaultParams.DEVICE,
    ):
        """
        Args:
            model:  已训练的 Seq2Seq 模型
            vocab:  词表映射，用于 encode 输入和 decode 输出
            device: 计算设备
        """
        self.model = model
        self.vocab = vocab
        self.device = torch.device(device) if isinstance(device, str) else device

        # 推理时固定为 eval 模式：
        # - 关闭 dropout（不再随机丢弃神经元）
        # - 关闭 batch norm 的 running mean/var 更新
        self.model.eval()
        self.model.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        device: torch.device | str = DefaultParams.DEVICE,
    ) -> "Predictor":
        """
        从 checkpoint 文件构建 Predictor

        自动加载模型权重、重建词表，完成推理就绪状态。

        步骤：
        1. 加载 checkpoint 字典
        2. 从 checkpoint["vocab"] 重建 VocabMapping
        3. 根据词表大小构建模型结构
        4. 加载模型权重

        Args:
            checkpoint_path: checkpoint .pth 文件路径
            device:          计算设备

        Returns:
            推理就绪的 Predictor 实例
        """
        device_obj = torch.device(device) if isinstance(device, str) else device

        # 先加载 checkpoint 获取词表信息（无需模型实例）
        checkpoint = torch.load(
            checkpoint_path, map_location=device_obj, weights_only=False
        )

        # 重建词表：checkpoint 中保存的是 to_dict() 的产物
        vocab = VocabMapping.from_dict(checkpoint["vocab"])

        # 构建与训练时结构一致的模型
        # 关键：词表大小必须与训练时相同，否则 embedding 维度不匹配
        model = build_model(
            vocab_size=vocab.vocabulary_size,
            pad_index=vocab.pad_index,
            sos_index=vocab.sos_index,
            eos_index=vocab.eos_index,
            device=device_obj,
        )

        # 加载训练好的权重
        model.load_state_dict(checkpoint["model_state_dict"])

        return cls(model, vocab, device_obj)

    # ==================================================================
    # 单条预测
    # ==================================================================
    def predict(self, expression: str) -> str:
        """
        对单条表达式进行推理

        内部流程：
        expression -> vocab.encode() -> tensor -> model.generate() -> vocab.decode()

        Args:
            expression: 输入表达式字符串，如 "35+27" 或 "99-50"

        Returns:
            计算结果字符串，如 "62" 或 "49"
        """
        return self.predict_batch([expression])[0]

    # ==================================================================
    # 批量预测
    # ==================================================================
    def predict_batch(self, expressions: List[str]) -> List[str]:
        """
        对多条表达式进行批量推理

        将字符串列表一次性 encode 为张量 batch，
        利用 GPU 并行加速生成。

        特殊处理：
        - 不同长度的表达式在 batch 维度做 padding（右侧填 PAD）
        - 生成序列通过 vocab.decode() 还原为字符串，
          strip_special=True 自动去除 PAD/SOS/EOS

        Args:
            expressions: 输入表达式字符串列表

        Returns:
            计算结果字符串列表，顺序与输入一一对应
        """
        if not expressions:
            return []

        # ================================================================
        # Step 1: 编码所有表达式
        # 每条表达式 encode 后长度可能不同，需 padding 到 max_len
        # ================================================================
        encoded_sequences = [
            self.vocab.encode(expression, add_sos_eos=False)
            for expression in expressions
        ]
        max_length = max(len(sequence) for sequence in encoded_sequences)

        # 创建 batch tensor，全填 PAD
        batch_size = len(encoded_sequences)
        encoder_input = torch.full(
            (batch_size, max_length),
            self.vocab.pad_index,
            dtype=torch.long,
            device=self.device,
        )
        encoder_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.bool,
            device=self.device,
        )

        for i, sequence in enumerate(encoded_sequences):
            sequence_length = len(sequence)
            encoder_input[i, :sequence_length] = torch.tensor(
                sequence, dtype=torch.long
            )
            encoder_mask[i, :sequence_length] = True

        # ================================================================
        # Step 2: 模型生成
        # InferenceParams.MAX_GEN_LENGTH=10 足够覆盖所有 0~99 以内的结果
        # 最长的结果是 "99+99=198"（3 个字符），加 SOS/EOS 共 5 个 token
        # ================================================================
        with torch.no_grad():
            generated_ids, _ = self.model.generate(
                encoder_input,
                encoder_mask,
                max_generation_length=InferenceParams.MAX_GEN_LENGTH,
            )

        # ================================================================
        # Step 3: 解码为字符串
        # ================================================================
        results = []
        for i in range(batch_size):
            result_string = self.vocab.decode(
                generated_ids[i].tolist(),
                strip_special=True,
            )
            results.append(result_string)

        return results

    # ==================================================================
    # 文件预测
    # ==================================================================
    def predict_file(
        self,
        input_path: Path,
        output_path: Path,
        expression_column: str = "expression",
    ) -> Path:
        """
        对 CSV 文件中指定列的表达式批量推理，结果写入新 CSV

        输出 CSV 包含原始列 + prediction 列。
        如果原始 CSV 有 output 列（真实答案），会额外计算并打印准确率。

        Args:
            input_path:         输入 CSV 文件路径
            output_path:        输出 CSV 文件路径
            expression_column:  表达式所在的列名，默认 "expression"

        Returns:
            输出文件路径
        """
        # 读取输入文件
        with input_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field_names = reader.fieldnames or []
            rows = list(reader)

        # 提取表达式列
        expressions = [row[expression_column].strip() for row in rows]

        # 批量推理
        predictions = self.predict_batch(expressions)

        # 写入输出文件：原始列 + prediction
        output_field_names = list(field_names) + ["prediction"]
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_field_names)
            writer.writeheader()
            for row, prediction in zip(rows, predictions):
                row["prediction"] = prediction
                writer.writerow(row)

        # 如果原始文件有 output 列，计算并打印准确率
        if "output" in field_names:
            targets = [row["output"].strip() for row in rows]
            correct = sum(
                1
                for prediction, target in zip(predictions, targets)
                if prediction == target
            )
            total = len(targets)
            accuracy = correct / total if total > 0 else 0.0
            print(f"  文件推理完成: {input_path}")
            print(f"  样本数: {total}")
            print(f"  正确数: {correct}")
            print(f"  准确率: {accuracy * 100:.2f}%")

        return output_path
