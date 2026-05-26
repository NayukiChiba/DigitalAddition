"""
模型模块

提供模型注册表、build_model 工厂函数，
以及 Encoder / Decoder / Seq2Seq 的导出。

模型注册表按 RNN 类型名称映射到构建函数，
支持通过字符串 "LSTM" / "RNN" / "GRU" 切换 Encoder 和 Decoder 的 RNN 类型。
"""

from typing import Dict

import torch

from config.defaults import DefaultParams, ModelParams
from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.seq2seq import Seq2Seq

# 模型注册表：RNN 类型名称，用于校验 rnn_type 参数
# Encoder 和 Decoder 内部通过 rnn_type 字符串选择 nn.LSTM / nn.RNN / nn.GRU
MODEL_REGISTRY: Dict[str, str] = {
    "RNN": "RNN",
    "LSTM": "LSTM",
    "GRU": "GRU",
}


def build_model(
    vocab_size: int,
    pad_index: int = 0,
    sos_index: int = 2,
    eos_index: int = 3,
    # --- Encoder 参数 ---
    encoder_embedding_dim: int = ModelParams.ENCODER_EMBEDDING_DIM,
    encoder_num_layers: int = ModelParams.ENCODER_NUM_LAYERS,
    encoder_bidirectional: bool = ModelParams.BIDIRECTIONAL,
    # --- Decoder 参数 ---
    decoder_embedding_dim: int = ModelParams.DECODER_EMBEDDING_DIM,
    decoder_num_layers: int = ModelParams.DECODER_NUM_LAYERS,
    # --- 共享参数（Encoder 与 Decoder 必须一致） ---
    hidden_dim: int = ModelParams.HIDDEN_DIM,
    rnn_type: str = ModelParams.RNN_TYPE,
    dropout: float = ModelParams.DROPOUT,
    # --- 设备 ---
    device: torch.device | None = DefaultParams.DEVICE,
) -> Seq2Seq:
    """
    模型工厂函数

    根据传入的超参数构建 Encoder 和 Decoder，封装为 Seq2Seq 模型。
    所有参数均有默认值（来自 ModelParams），调用方可按需覆盖。

    hidden_dim、rnn_type、dropout 在 Encoder 和 Decoder 之间必须一致，
    因为 Encoder 的 hidden state 直接作为 Decoder 的初始状态，
    维度不匹配会导致运行时错误。

    Args:
        vocab_size:             词表大小（16）
        pad_index:              PAD 标记索引，默认 0
        sos_index:              SOS 起始标记索引，默认 2
        eos_index:              EOS 终止标记索引，默认 3
        encoder_embedding_dim:  Encoder 词嵌入维度，默认 128
        encoder_num_layers:     Encoder RNN 层数，默认 2
        encoder_bidirectional:  Encoder 是否双向，默认 False
        decoder_embedding_dim:  Decoder 词嵌入维度，默认 128
        decoder_num_layers:     Decoder RNN 层数，默认 2
        hidden_dim:             RNN 隐藏层维度（Encoder/Decoder 共享），默认 256
        rnn_type:               RNN 类型 "LSTM" / "RNN" / "GRU"，默认 "LSTM"
        dropout:                Dropout 概率，默认 0.3
        device:                 计算设备，默认自动选择 cuda / cpu

    Returns:
        Seq2Seq 实例，可直接用于训练和推理

    Raises:
        ValueError: 当 rnn_type 不在 MODEL_REGISTRY 中时抛出
    """
    if rnn_type not in MODEL_REGISTRY:
        raise ValueError(
            f"未知的 RNN 类型: '{rnn_type}'，请从 {list(MODEL_REGISTRY.keys())} 中选择"
        )

    # 构建 Encoder：将输入表达式字符序列编码为上下文向量
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=encoder_embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=encoder_num_layers,
        rnn_type=rnn_type,
        dropout=dropout,
        pad_index=pad_index,
        bidirectional=encoder_bidirectional,
    )

    # 构建 Decoder：从上下文向量自回归解码为目标字符序列（始终单向）
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=decoder_embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=decoder_num_layers,
        rnn_type=rnn_type,
        dropout=dropout,
        pad_index=pad_index,
    )

    # 封装为 Seq2Seq：串联 Encoder + Decoder，提供训练与推理接口
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        vocab_size=vocab_size,
        pad_index=pad_index,
        sos_index=sos_index,
        eos_index=eos_index,
        device=device,
    )

    return model


__all__ = [
    "Encoder",
    "Decoder",
    "Seq2Seq",
    "MODEL_REGISTRY",
    "build_model",
]
