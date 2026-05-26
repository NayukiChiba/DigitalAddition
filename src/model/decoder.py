"""
Decoder 解码器模块(无 Attention)

从 Encoder 的上下文向量出发,自回归地逐字符生成目标序列.
训练时使用 Teacher Forcing,推理时使用贪心解码.

架构:Embedding → Dropout → RNN → Linear(→ vocab_size)
"""

from typing import Literal, Tuple

import torch
import torch.nn as nn

from config.defaults import ModelParams


class Decoder(nn.Module):
    """
    Seq2Seq 解码器(无 Attention)

    以 Encoder 的 final hidden state 初始化 RNN 的隐藏状态,
    逐时间步生成下一个字符的 logits.

    无 Attention 意味着 Decoder 在每个时间步都只能间接访问输入信息
    ——只有 Encoder 压缩进 hidden state 的那部分能传递过来.
    所以 Encoder 的 hidden state 被称作"信息瓶颈".

    对于 100 以内加减法,输入最长 5 字符(如 "99+99"),
    输出最长 3 字符(最大结果 198),信息量不大,
    无 Attention 的架构理论上足够.

    Args:
        vocab_size: 词表大小(16,含特殊 token)
        embedding_dim: 词嵌入向量维度
        hidden_dim: RNN 隐藏层维度(需与 Encoder 的 hidden_dim 一致)
        num_layers: RNN 堆叠层数(需与 Encoder 的 num_layers 一致,因为 Encoder 的 hidden state 直接用作 Decoder 的初始状态)
        rnn_type: RNN 类型,"LSTM" / "RNN" / "GRU"
        dropout: Dropout 概率
        pad_index: PAD 标记索引
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = ModelParams.DECODER_EMBEDDING_DIM,
        hidden_dim: int = ModelParams.HIDDEN_DIM,
        num_layers: int = ModelParams.DECODER_NUM_LAYERS,
        rnn_type: Literal["LSTM", "RNN", "GRU"] = ModelParams.RNN_TYPE,
        dropout: float = ModelParams.DROPOUT,
        pad_index: int = 0,
    ):
        super().__init__()

        # --- Embedding 层 ---
        # 将 decoder 输入的字符索引(如 [SOS, 6, 2])映射为向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

        self.dropout = nn.Dropout(dropout)
        self.rnn_type = rnn_type

        # --- RNN 层 ---
        # 结构与 Encoder 对应,但始终单向(解码是自回归的,不能看未来)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_class = {"LSTM": nn.LSTM, "RNN": nn.RNN, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_class(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            bidirectional=False,  # 解码器必须是单向的
            batch_first=True,
        )

        # --- 输出投影层 ---
        # 将 RNN 的隐藏状态映射到词表空间,得到每个字符的预测分数
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播(训练模式,单步或全序列)

        Args:
            input_ids: 解码器输入索引,形状 (batch_size, seq_len)
                       训练时为 decoder_input = [SOS, 6, 2]
                       推理时每步传入单个 token,形状 (batch_size, 1)
            hidden:    从 Encoder 传递来的初始隐藏状态
                       - LSTM: (h_n, c_n)
                       - RNN / GRU: h_n

        Returns:
            logits: 每个时间步的预测分数,形状 (batch_size, seq_len, vocab_size)
                    未归一化,后续由 CrossEntropyLoss 处理
            hidden: 更新后的隐藏状态(可用于下一步自回归生成)
        """
        # Step 1: Embedding + Dropout
        # (B, S) → (B, S, embedding_dim)
        embedded = self.dropout(self.embedding(input_ids))

        # Step 2: RNN 前向
        # outputs: (B, S, hidden_dim)
        # hidden: 更新后的 final hidden state
        outputs, hidden = self.rnn(embedded, hidden)

        # Step 3: 线性投影到词表空间
        # (B, S, hidden_dim) → (B, S, vocab_size)
        logits = self.output_layer(outputs)

        return logits, hidden
