"""
评估指标模块

提供 Seq2Seq 算术任务的专用评估指标。

与分类任务的区别：
- 分类任务用 accuracy/precision/recall/f1，每个样本一个标签
- Seq2Seq 生成任务用 Exact Match + Token Accuracy，
  前者衡量整条序列是否正确，后者衡量每个位置字符是否正确

指标选择理由：
- Exact Match（完全匹配）：最严格的指标，生成序列与目标序列逐 token 一致才算正确
  这是算术任务的最终目标——答案必须完全正确才有意义
- Token Accuracy（字符级准确率）：宽松指标，反映模型在每个位置上的预测能力
  即使最终答案不对，也能看出模型在哪个位置容易出错
- 不计算 BLEU/ROUGE：这些是 NLP 文本生成指标，对算术任务无意义
"""

from typing import Tuple

import torch


def compute_exact_match(
    generated_ids: torch.Tensor,
    target_output: torch.Tensor,
    eos_index: int = 3,
) -> Tuple[int, int]:
    """
    计算完全匹配准确率

    逐样本比较生成序列与目标序列，所有非 PAD 位置全部一致才算正确。
    EOS 之后的填充位置不参与比较。

    Args:
        generated_ids: 模型生成的 token 序列，形状 [batch_size, max_gen_len]
        target_output: 真实目标序列，形状 [batch_size, target_len]
        eos_index: EOS 标记索引，用于截断比较范围

    Returns:
        (correct, total): 完全正确的样本数, 总样本数
    """
    batch_size = target_output.size(0)
    correct = 0

    for i in range(batch_size):
        target_sequence = target_output[i]

        # 目标序列末尾可能有 PAD（批次内长度不一致时的填充）
        # 找到有效长度：EOS 之后的都是 PAD，不参与比较
        eos_mask = target_sequence == eos_index
        if eos_mask.any():
            # EOS 出现的位置 + 1 = 有效长度（含 EOS 本身）
            target_length = eos_mask.nonzero(as_tuple=True)[0][0].item() + 1
            target_sequence = target_sequence[:target_length]

        # 生成序列截取与目标等长的部分
        prediction_sequence = generated_ids[i, :target_length]

        if torch.equal(prediction_sequence, target_sequence):
            correct += 1

    return correct, batch_size


def compute_token_accuracy(
    generated_ids: torch.Tensor,
    target_output: torch.Tensor,
    pad_index: int = 0,
    eos_index: int = 3,
) -> Tuple[int, int]:
    """
    计算字符级准确率

    逐位置比较预测 token 与目标 token，忽略 PAD 位置。
    反映模型在每个字符位置上的预测能力，比 Exact Match 更细粒度。

    Args:
        generated_ids: 模型生成的 token 序列，形状 [batch_size, max_gen_len]
        target_output: 真实目标序列，形状 [batch_size, target_len]
        pad_index: PAD 标记索引，这些位置不参与比较
        eos_index: EOS 标记索引，EOS 之后的位置不参与比较

    Returns:
        (correct_tokens, total_tokens): 正确的 token 数, 总有效 token 数
    """
    batch_size = target_output.size(0)
    total_correct = 0
    total_tokens = 0

    for i in range(batch_size):
        target_sequence = target_output[i]
        prediction_sequence = generated_ids[i, : target_sequence.size(0)]

        for j in range(target_sequence.size(0)):
            # 只比较非 PAD 位置
            if target_sequence[j].item() == pad_index:
                continue
            total_tokens += 1
            if prediction_sequence[j].item() == target_sequence[j].item():
                total_correct += 1

    return total_correct, total_tokens


def compute_metrics(
    generated_ids: torch.Tensor,
    target_output: torch.Tensor,
    pad_index: int = 0,
    eos_index: int = 3,
) -> dict:
    """
    计算全部评估指标

    一次性计算 Exact Match 和 Token Accuracy，返回字典。

    Args:
        generated_ids: 模型生成的 token 序列，形状 [batch_size, max_gen_len]
        target_output: 真实目标序列，形状 [batch_size, target_len]
        pad_index: PAD 标记索引
        eos_index: EOS 标记索引

    Returns:
        {
            "exact_match": float,       # 完全匹配准确率 [0, 1]
            "exact_match_correct": int, # 完全正确的样本数
            "exact_match_total": int,   # 总样本数
            "token_accuracy": float,    # 字符级准确率 [0, 1]
            "token_correct": int,       # 正确的 token 数
            "token_total": int,         # 总有效 token 数
        }
    """
    exact_match_correct, exact_match_total = compute_exact_match(
        generated_ids, target_output, eos_index
    )
    token_correct, token_total = compute_token_accuracy(
        generated_ids, target_output, pad_index, eos_index
    )

    return {
        "exact_match": exact_match_correct / exact_match_total
        if exact_match_total > 0
        else 0.0,
        "exact_match_correct": exact_match_correct,
        "exact_match_total": exact_match_total,
        "token_accuracy": token_correct / token_total if token_total > 0 else 0.0,
        "token_correct": token_correct,
        "token_total": token_total,
    }
