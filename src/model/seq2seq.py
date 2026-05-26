"""
Seq2Seq 模型封装模块

将 Encoder 和 Decoder 串联为完整的序列到序列模型.
提供训练前向(含 Teacher Forcing)和推理生成(贪心解码)两种接口.
"""

import random
from typing import Tuple

import torch
import torch.nn as nn

from config.defaults import DefaultParams, ModelParams
from src.model.decoder import Decoder
from src.model.encoder import Encoder


class Seq2Seq(nn.Module):
    """
    Seq2Seq 模型(无 Attention)

    完整流程:
    训练时: Encoder 编码输入 -> Teacher Forcing 驱动 Decoder -> 输出 logits
    推理时: Encoder 编码输入 -> 贪心解码自回归生成 -> 输出 token 序列

    Teacher Forcing 原理:
    在每个解码时间步,以概率 teacher_forcing_ratio 使用真实的目标 token
    而不是模型自己预测的 token 作为下一时间步的输入.
    这能加速收敛,防止模型在训练早期因错误级联而难以学习.
    推理时不存在 Teacher Forcing,完全依赖模型自身的预测.

    Args:
        encoder: Encoder 实例
        decoder: Decoder 实例
        vocab_size: 词表大小
        pad_index: PAD 标记索引(用于损失计算时忽略 PAD 位置)
        sos_index: SOS 起始标记索引
        eos_index: EOS 终止标记索引
        device: 计算设备
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        vocab_size: int,
        pad_index: int = 0,
        sos_index: int = 2,
        eos_index: int = 3,
        device: torch.device | None = DefaultParams.DEVICE,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.pad_index = pad_index
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.device = device

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        teacher_forcing_ratio: float = ModelParams.TEACHER_FORCING_RATIO,
    ) -> torch.Tensor:
        """
        训练前向传播

        先通过 Encoder 编码输入序列得到上下文向量,
        再用 Teacher Forcing 驱动 Decoder 生成每个时间步的 logits.

        Args:
            encoder_input: 编码器输入索引,
                           形状 [batch_size, encoder_length]
                           例:[[7, 9, 14, 6, 11]] 表示 "35+27"
            decoder_input: 解码器输入索引(含 SOS 不含 EOS),
                           形状 [batch_size, decoder_length]
                           例:[[2, 10, 6]] 表示 "<SOS>62"
            encoder_mask: 编码器有效位置 mask,
                          形状 [batch_size, encoder_length]
                          True=有效字符,False=PAD 填充
            teacher_forcing_ratio: Teacher Forcing 概率
                                   1.0 = 每步都用真实 token
                                   0.0 = 每步都用自身预测 token

        Returns:
            logits: 每个时间步的预测分数(未归一化),
                    形状 [batch_size, decoder_length, vocab_size]
                    与 target_output 做交叉熵,
                    ignore_index=pad_index 忽略 PAD 位置
        """
        decoder_length = decoder_input.size(1)

        # ============================================================
        # Step 1: Encoder 编码
        # encoder_outputs: [batch_size, encoder_length, hidden_dim]
        #   无 Attention 时不直接使用 outputs,只取 hidden state
        # encoder_hidden: 最后时刻的隐藏状态,用作 Decoder 初始状态
        #   - LSTM: (h_n, c_n) 元组
        #   - RNN / GRU: h_n 张量
        #   各分量形状: [num_layers * num_directions, batch_size, hidden_dim]
        # ============================================================
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_mask)

        # ============================================================
        # Step 2: 准备 Decoder 的初始输入与初始隐藏状态
        # 取 decoder_input 的第一列 <SOS> token 作为第一个时间步的输入
        # decoder_input: [batch_size, decoder_length] -> [batch_size, 1]
        # ============================================================
        current_input = decoder_input[:, 0].unsqueeze(1)
        hidden_state = encoder_hidden

        # ============================================================
        # Step 3: 逐时间步解码
        # 每个时间步:当前 token + 隐藏状态 -> Decoder -> 下一 token 的 logits
        # Teacher Forcing 决策在每个时间步末尾进行:
        #   - 以概率 teacher_forcing_ratio 取 decoder_input 中的真实 token
        #   - 否则取模型当前预测的 token(贪心 argmax)
        # ============================================================
        logits_list = []

        for step in range(decoder_length):
            # 单步解码:输入 [batch_size, 1],输出 [batch_size, 1, vocab_size]
            step_output, hidden_state = self.decoder(current_input, hidden_state)
            logits_list.append(step_output)

            # 最后一步不需要准备下一输入
            if step == decoder_length - 1:
                break

            # --- Teacher Forcing 决策 ---
            if random.random() < teacher_forcing_ratio:
                # 使用真实的下一个 token
                # [batch_size, decoder_length] -> [batch_size, 1]
                current_input = decoder_input[:, step + 1].unsqueeze(1)
            else:
                # 使用模型当前预测的 token(贪心 argmax)
                # [batch_size, 1, vocab_size] -> [batch_size] -> [batch_size, 1]
                predicted_token = step_output.squeeze(1).argmax(dim=1)
                current_input = predicted_token.unsqueeze(1)

        # ============================================================
        # Step 4: 拼接所有时间步的 logits
        # 列表中每个元素: [batch_size, 1, vocab_size]
        # 沿时间维拼接 -> [batch_size, decoder_length, vocab_size]
        # logits[step] 的监督目标是 target_output[step](即 decoder_input 右移一位)
        # ============================================================
        logits = torch.cat(logits_list, dim=1)

        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_input: torch.Tensor,
        encoder_mask: torch.Tensor | None = None,
        max_generation_length: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推理生成(贪心解码)

        Encoder 编码后,从 <SOS> 开始自回归生成,
        每步选取概率最高的 token,直到遇到 <EOS> 或达到最大生成长度.

        与训练时的区别:
        - 全程关闭梯度(@torch.no_grad)
        - 不使用 Teacher Forcing,每步用自身预测作为下一输入
        - 遇到 <EOS> 后停止该样本的生成(其余样本继续)

        Args:
            encoder_input: 编码器输入索引,
                           形状 [batch_size, encoder_length]
            encoder_mask: 编码器有效位置 mask,
                          形状 [batch_size, encoder_length]
            max_generation_length: 最大生成长度,防止无限循环

        Returns:
            generated_ids: 生成的 token 索引序列,
                           形状 [batch_size, max_generation_length]
                           提前结束的样本在 <EOS> 之后填充 pad_index
            sequence_lengths: 每条序列的实际生成长度(含 <EOS>),
                              形状 [batch_size]
        """
        batch_size = encoder_input.size(0)

        # ============================================================
        # Step 1: Encoder 编码
        # 不需要 encoder_outputs,只取 hidden state 作为 Decoder 初始状态
        # ============================================================
        _, encoder_hidden = self.encoder(encoder_input, encoder_mask)

        # ============================================================
        # Step 2: 初始化生成状态
        # ============================================================
        # 每个样本的首个输入固定为 <SOS> token
        current_input = torch.full(
            (batch_size, 1),
            self.sos_index,
            dtype=torch.long,
            device=self.device,
        )
        hidden_state = encoder_hidden

        # 预分配输出张量:全填 pad_index,逐步写入预测结果
        generated_ids = torch.full(
            (batch_size, max_generation_length),
            self.pad_index,
            dtype=torch.long,
            device=self.device,
        )
        sequence_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # 标记每个样本是否已生成完毕(遇到 <EOS> 即置 True)
        is_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # ============================================================
        # Step 3: 自回归逐时间步生成
        # ============================================================
        for step in range(max_generation_length):
            # 单步解码:输入当前 token,输出下一 token 的 logits
            # step_output: [batch_size, 1, vocab_size]
            step_output, hidden_state = self.decoder(current_input, hidden_state)

            # 贪心解码:取概率最高的 token
            # [batch_size, 1, vocab_size] -> [batch_size, vocab_size] -> [batch_size]
            predicted_token = step_output.squeeze(1).argmax(dim=1)

            # 将当前步预测写入输出
            generated_ids[:, step] = predicted_token

            # 首次遇到 <EOS> 的样本,记录其生成长度(含 <EOS> 本身)
            just_finished = (predicted_token == self.eos_index) & ~is_finished
            sequence_lengths[just_finished] = step + 1
            is_finished = is_finished | (predicted_token == self.eos_index)

            # 所有样本都已结束,提前退出循环
            if is_finished.all():
                break

            # 当前预测作为下一时间步的输入
            # [batch_size] -> [batch_size, 1]
            current_input = predicted_token.unsqueeze(1)

        # 从未遇到 <EOS> 的样本,长度为 max_generation_length
        sequence_lengths[~is_finished] = max_generation_length

        return generated_ids, sequence_lengths
