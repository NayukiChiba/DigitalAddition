"""
默认超参数配置

- DefaultParams: 全局参数(随机种子、设备)
- DataParams: 数据生成与加载参数
- ModelParams: Seq2Seq 模型结构参数(无 Attention)
- TrainingParams: 训练相关参数
- InferenceParams: 推理相关参数
"""

from typing import Literal

import torch


class DefaultParams:
    # 全局随机种子
    RANDOM_SEED = 42

    # 设备配置(自动选择 GPU 或 CPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DataParams:
    """
    数据生成与加载参数

    """

    # 运算数字范围
    MIN_NUMBER = 0
    MAX_NUMBER = 100

    # 允许的运算符
    OPERATORS = ["+", "-"]

    # 生成样本数量
    TRAIN_SIZE = 50000
    VAL_SIZE = 5000
    TEST_SIZE = 5000

    # DataLoader 参数
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    PIN_MEMORY = True  # 是否将数据集加载到内存中以加速训练
    SHUFFLE = True  # 是否在每个 epoch 结束后打乱数据

    # 输入/输出序列最大长度, None 表示不限制长度(根据实际数据自动调整)
    MAX_INPUT_LENGTH = None
    MAX_OUTPUT_LENGTH = None

    # 词表最小频率
    MIN_FREQ = 1


class ModelParams:
    """Seq2Seq 模型结构参数(无 Attention)"""

    RNN_TYPE: Literal["LSTM", "RNN", "GRU"] = "LSTM"  # RNN 类型(LSTM 或 GRU)

    # 嵌入维度
    ENCODER_EMBEDDING_DIM = 128
    DECODER_EMBEDDING_DIM = 128

    # 隐藏层维度(Encoder 与 Decoder 共享)
    HIDDEN_DIM = 256

    # RNN 层数
    ENCODER_NUM_LAYERS = 2
    DECODER_NUM_LAYERS = 2

    # Encoder 是否使用双向 RNN
    BIDIRECTIONAL = False

    # Teacher Forcing 概率
    # 在训练过程中,教师强制(Teacher Forcing)是一种常用的技术
    # 它在训练解码器时使用真实的目标输出作为输入,而不是模型的预测输出.
    # TEACHER_FORCING_RATIO 定义了在每个时间步使用教师强制的概率.
    # 例如,如果 TEACHER_FORCING_RATIO 设置为 0.5,那么在训练过程中有 50% 的时间步会使用真实的目标输出
    # 另 50% 的时间步会使用模型的预测输出.这有助于模型更快地收敛,同时也能提高模型在推理阶段的性能.
    TEACHER_FORCING_RATIO = 0.5

    # RNN Dropout(层数 > 1 时生效)
    DROPOUT = 0.3


class TrainingParams:
    """训练相关参数"""

    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    EPOCHS = 50
    GRAD_CLIP = 5.0

    OPTIMIZER: Literal["Adam", "SGD", "AdamW"] = "Adam"
    WEIGHT_DECAY = 1e-4

    LR_SCHEDULER: Literal["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"] = "StepLR"
    LR_STEP_SIZE = 10
    LR_GAMMA = 0.5
    LR_REDUCE_FACTOR = 0.5
    LR_REDUCE_PATIENCE = 3

    # 早停
    EARLY_STOP_PATIENCE = 5
    EARLY_STOP_MIN_DELTA = 1e-4

    LOG_INTERVAL = 50
    CHECKPOINT_INTERVAL = 5


class InferenceParams:
    """推理相关参数"""

    MAX_GEN_LENGTH = 10
    TEMPERATURE = 0.8
    TOP_K = 3
    # TOP_P (nucleus sampling) 是一种在文本生成中使用的采样方法,
    # 它通过选择概率总和达到某个阈值 p 的前 k 个候选词来限制生成的词汇范围.
    # 与 Top-K 采样不同,Top-P 采样不固定选择前 k 个词,而是根据概率分布动态选择词汇,
    # 从而更灵活地控制生成文本的多样性和质量.
    TOP_P = 0.9
