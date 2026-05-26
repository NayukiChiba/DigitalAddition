"""
Encoder 编码器模块

将输入字符序列(如 "35+27")编码为上下文向量.
通过 RNN 逐字符处理,取最后时刻的 hidden state 作为整个输入序列的语义表征,
传递给 Decoder 作为初始隐藏状态.

支持 SimpleRNN / LSTM / GRU 三种 RNN 类型,通过参数切换.
"""

from typing import Literal

import torch
import torch.nn as nn

# pack_padded_sequence 是 PyTorch 提供的工具函数,用于处理变长序列的 RNN 输入
# padded_packed_sequence 则是它的逆操作,将 RNN 输出的 PackedSequence 转回普通张量
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config.defaults import ModelParams


class Encoder(nn.Module):
    """
    Seq2Seq 编码器

    架构:Embedding → Dropout → RNN
    输出 RNN 全部时刻的 outputs 和最后时刻的 hidden state.

    不使用 Attention,因此 Decoder 仅依赖 final hidden state 作为上下文.
    这对 Encoder 提出了更高要求:必须将整个表达式的语义(数字、运算符、计算逻辑)压缩到一个固定维度的向量中.

    Args:
        vocab_size: 词表大小(16,含特殊 token)
        embedding_dim: 词嵌入向量维度
        hidden_dim: RNN 隐藏层维度
        num_layers: RNN 堆叠层数
        rnn_type: RNN 类型,"LSTM" / "RNN" / "GRU"
        dropout: Dropout 概率,仅当 num_layers > 1 时作用于层间
        pad_index: PAD 标记的索引,Embedding 层对该位置输出零向量
        bidirectional: 是否使用双向 RNN(本项目默认 False)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = ModelParams.ENCODER_EMBEDDING_DIM,
        hidden_dim: int = ModelParams.HIDDEN_DIM,
        num_layers: int = ModelParams.ENCODER_NUM_LAYERS,
        rnn_type: Literal["LSTM", "RNN", "GRU"] = ModelParams.RNN_TYPE,
        dropout: float = ModelParams.DROPOUT,
        pad_index: int = 0,
        bidirectional: bool = ModelParams.BIDIRECTIONAL,
    ):
        super().__init__()

        # Embedding 层: 将输入的字符索引序列转换为嵌入向量序列
        # padding_idx=pad_index 参数确保 PAD 索引位置的嵌入向量为零,不参与训练更新
        # 不受梯度更新的 PAD 向量在训练过程中保持为零,有助于模型区分有效输入和填充部分
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )

        # Dropout 层: 随机丢弃部分嵌入向量,防止过拟合
        # 在 Embedding 层后使用 Dropout 防止过拟合,尤其是在训练数据较少或模型较大时
        self.dropout = nn.Dropout(dropout)

        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 双向翻倍, 单向为1, 用于统一 output 维度计算
        self.num_directions = 2 if bidirectional else 1

        # RNN 层: 根据 rnn_type 参数选择 RNN 类型
        # 注意: PyTorch 的 RNN 在 num_layers > 1 时会在层与层之间自动添加 Dropout, 无需手动添加额外的 Dropout 层
        rnn_dropout = dropout if num_layers > 1 else 0.0

        # 根据 rnn_type 参数选择 RNN 类型,并设置相应的输入输出维度
        rnn_cls = {"LSTM": nn.LSTM, "RNN": nn.RNN, "GRU": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=embedding_dim,  # RNN 输入维度等于嵌入维度
            hidden_size=hidden_dim,  # RNN 隐藏层维度
            num_layers=num_layers,  # RNN 层数
            dropout=rnn_dropout,  # RNN Dropout,仅在层数 > 1 时生效
            bidirectional=bidirectional,  # 是否使用双向 RNN
            batch_first=True,  # 输入输出张量形状为 (batch, seq, feature)
        )

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids(torch.Tensor): 编码器输入索引,形状 (batch_size, seq_len)
                       例:[[7, 9, 14, 6, 11]] 表示 "35+27"
            mask(torch.Tensor | None): 有效位置掩码,形状 (batch_size, seq_len)
                  True 表示有效字符,False 表示 PAD 填充.
                  传入时启用 pack_padded_sequence 跳过无效时间步加速计算.

        Returns:
            outputs: RNN 所有时刻的输出，
                     形状 (batch_size, seq_len, hidden_dim * num_directions)
            hidden:  最后时刻的隐藏状态，用于初始化 Decoder
                     - LSTM: (h_n, c_n)，各形状为
                       (num_layers * num_directions, batch_size, hidden_dim)
                     - RNN / GRU: h_n，形状同上
        """
        # Step 1: Embedding + Dropout
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(embedded)

        # Step 2: RNN 前向传播
        if mask is not None:
            # 有 mask 时使用 pack_padded_sequence 优化：
            # 将变长序列打包为紧凑格式，跳过 PAD 位置，提升计算效率
            seq_lengths = mask.sum(dim=1).cpu()  # 每条序列的实际长度
            packed = pack_padded_sequence(
                embedded, seq_lengths, batch_first=True, enforce_sorted=False
            )
            packed_outputs, hidden = self.rnn(packed)
            # 解包回 padded 格式，PAD 位置自动填 0
            outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            # 无 mask 时直接前向（批次内序列等长时可用）
            outputs, hidden = self.rnn(embedded)

        return outputs, hidden
