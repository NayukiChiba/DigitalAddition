"""
模型模块

提供模型注册表、build_model 工厂函数，
以及 Encoder / Decoder / Seq2Seq 的导出。

模型注册表按 RNN 类型名称映射到构建函数，
支持通过字符串 "LSTM" / "RNN" / "GRU" 切换 Encoder 和 Decoder 的 RNN 类型。
"""

from typing import Any, Dict

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.model.seq2seq import Seq2Seq

# 模型注册表：RNN 类型名称 → (Encoder类, Decoder类)
# 当前三种 RNN 共享同一套 Encoder/Decoder 类，通过 rnn_type 参数区分，
# 注册表主要用于 build_model 的接口统一，后续可扩展为不同的子类实现
MODEL_REGISTRY: Dict[str, Any] = {
    "rnn": "RNN",
    "lstm": "LSTM",
    "gru": "GRU",
}


def build_model(
    vocab_size: int,
    pad_index: int = 0,
    sos_index: int = 2,
    eos_index: int = 3,
    **kwargs,
) -> Seq2Seq:
    """
    模型工厂函数

    根据 kwargs 中的超参数构建 Encoder 和 Decoder，封装为 Seq2Seq 模型。

    参数优先级：kwargs > 类默认值
    即不传参数时使用 Encoder/Decoder 构造函数的默认值。

    Args:
        vocab_size: 词表大小（固定 16）
        pad_index:  PAD 标记索引（固定 0）
        sos_index:  SOS 标记索引（固定 2）
        eos_index:  EOS 标记索引（固定 3）
        **kwargs:   传递给 Encoder / Decoder 的超参数，包括：
            - embedding_dim (int): 词嵌入维度，默认 128
            - hidden_dim (int): 隐藏层维度，默认 256
            - num_layers (int): RNN 层数，默认 2
            - rnn_type (str): RNN 类型 "LSTM" / "RNN" / "GRU"，默认 "LSTM"
            - dropout (float): Dropout 概率，默认 0.3
            - bidirectional (bool): Encoder 是否双向，默认 False
            - device (torch.device): 计算设备

    Returns:
        Seq2Seq 实例，可直接用于训练和推理

    Raises:
        ValueError: 当 rnn_type 不在 MODEL_REGISTRY 中时抛出
    """
    # 从 kwargs 中提取参数，不存在则使用默认值
    rnn_type = kwargs.get("rnn_type", "LSTM")
    embedding_dim = kwargs.get("embedding_dim", 128)
    hidden_dim = kwargs.get("hidden_dim", 256)
    num_layers = kwargs.get("num_layers", 2)
    dropout = kwargs.get("dropout", 0.3)
    bidirectional = kwargs.get("bidirectional", False)
    device = kwargs.get("device", None)

    # 校验 RNN 类型
    if rnn_type not in MODEL_REGISTRY:
        raise ValueError(
            f"未知的 RNN 类型: '{rnn_type}'，请从 {list(MODEL_REGISTRY.keys())} 中选择"
        )

    # 构建 Encoder
    encoder = Encoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type=rnn_type,
        dropout=dropout,
        pad_index=pad_index,
        bidirectional=bidirectional,
    )

    # 构建 Decoder（始终单向）
    decoder = Decoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type=rnn_type,
        dropout=dropout,
        pad_index=pad_index,
    )

    # 封装为 Seq2Seq
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
