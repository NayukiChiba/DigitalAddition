"""
DataLoader 模块

提供 AdditionDataset 类和 create_data_loaders 工厂函数,
封装 PyTorch Dataset/DataLoader 的创建逻辑.

序列构造规则:
- Encoder 输入: [3, 5, +, 2, 7]   ← 纯字符索引,无 SOS/EOS
- Decoder 输入: [SOS, 6, 2]        ← 含 SOS,不含 EOS
- Target 输出: [6, 2, EOS]         ← 不含 SOS,含 EOS

通过 Shift 对齐:decoder_input[i] 的监督目标是 target_output[i],
即每个时间步预测下一个字符.交叉熵计算时 ignore_index=PAD_IDX.
"""

import csv
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config.defaults import DataParams
from config.paths import RAW_TEST_PATH, RAW_TRAIN_PATH, RAW_VAL_PATH
from src.data.mapping import VocabMapping


class AdditionDataset(Dataset):
    """
    加减法算式 Dataset

    加载 CSV 数据,通过 VocabMapping 将字符串 tokenize 为索引序列.

    每条 CSV 行 "35+27","62" 被处理为三组索引:
    - encoder_input:  [7, 9, 14, 6, 11]        ← "3""5""+""2""7" 的索引
    - decoder_input:  [2, 10, 6]               ← "<SOS>""6""2" 的索引
    - target_output:  [10, 6, 3]               ← "6""2""<EOS>" 的索引

    __getitem__ 返回 1D LongTensor 三元组,长度各异.
    """

    def __init__(self, csv_path: Path, vocab: VocabMapping):
        """
        Args:
            csv_path: CSV 文件路径(含 input/output 列)
            vocab: VocabMapping 实例
        """
        self.vocab = vocab
        self.samples: List[Tuple[List[int], List[int], List[int]]] = []

        # 在构造阶段一次性加载并 tokenize 全部样本,避免每个 epoch 重复转换
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_expr = row["input"].strip()
                output_result = row["output"].strip()

                # Encoder 输入:纯字符索引,不加 SOS/EOS
                encoder_input = vocab.encode(input_expr, add_sos_eos=False)

                # Decoder 输入:加 SOS/EOS,再去掉末尾 EOS -> 得到 [SOS, ...]
                decoder_input = vocab.encode(output_result, add_sos_eos=True)[:-1]

                # Target 输出:加 SOS/EOS,再去掉开头 SOS -> 得到 [..., EOS]
                target_output = vocab.encode(output_result, add_sos_eos=True)[1:]

                self.samples.append((encoder_input, decoder_input, target_output))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_input, decoder_input, target_output = self.samples[idx]
        return (
            torch.tensor(encoder_input, dtype=torch.long),
            torch.tensor(decoder_input, dtype=torch.long),
            torch.tensor(target_output, dtype=torch.long),
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批次整理函数

    对 encoder/decoder/target 三组序列分别做 padding 到批次内最大长度.
    PAD_IDX=0 作为填充值.

    Args:
        batch: [(encoder_input, decoder_input, target_output), ...]

    Returns:
        (encoder_input, decoder_input, target_output, encoder_mask)
        - encoder_input: [batch_size, max_encoder_len]  编码器输入,0 填充
        - decoder_input: [batch_size, max_decoder_len]  解码器输入,0 填充
        - target_output: [batch_size, max_target_len]   目标输出,0 填充
        - encoder_mask:  [batch_size, max_encoder_len]  bool 张量
                         True=有效位置,False=PAD
                         用于 RNN pack_padded_sequence
    """
    encoder_inputs, decoder_inputs, target_outputs = zip(*batch)

    # 批次内各序列长度不同,取各自最大值做 padding 基准
    max_encoder_len = max(s.size(0) for s in encoder_inputs)
    max_decoder_len = max(s.size(0) for s in decoder_inputs)
    max_target_len = max(s.size(0) for s in target_outputs)
    batch_size = len(batch)

    # 全量初始化为 PAD_IDX=0,再逐条填入实际序列
    padded_encoder = torch.zeros((batch_size, max_encoder_len), dtype=torch.long)
    padded_decoder = torch.zeros((batch_size, max_decoder_len), dtype=torch.long)
    padded_target = torch.zeros((batch_size, max_target_len), dtype=torch.long)
    encode_mask = torch.zeros((batch_size, max_encoder_len), dtype=torch.bool)

    for i, (enc, dec, tgt) in enumerate(
        zip(encoder_inputs, decoder_inputs, target_outputs)
    ):
        padded_encoder[i, : enc.size(0)] = enc
        padded_decoder[i, : dec.size(0)] = dec
        padded_target[i, : tgt.size(0)] = tgt
        encode_mask[i, : enc.size(0)] = True

    return padded_encoder, padded_decoder, padded_target, encode_mask


def create_data_loaders(
    vocab: VocabMapping,
    train_csv_path: Path = RAW_TRAIN_PATH,
    val_csv_path: Path = RAW_VAL_PATH,
    test_csv_path: Path = RAW_TEST_PATH,
    batch_size: int = DataParams.BATCH_SIZE,
    num_workers: int = DataParams.NUM_WORKERS,
    pin_memory: bool = DataParams.PIN_MEMORY,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练集、验证集和测试集的 DataLoader

    训练集设 shuffle=True 以打乱样本顺序,避免模型记忆数据排列;
    验证集和测试集设 shuffle=False,保证每次评估结果一致可复现.

    Args:
        train_csv_path(Path): 训练集 CSV 文件路径
        val_csv_path(Path):   验证集 CSV 文件路径
        test_csv_path(Path):  测试集 CSV 文件路径
        vocab(VocabMapping):          VocabMapping 实例
        batch_size(int):     批大小
        num_workers(int):    DataLoader 工作进程数
        pin_memory(bool):  是否启用 pin_memory(GPU 训练时建议开启)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 构建三个 Dataset(内部完成 tokenize)
    train_dataset = AdditionDataset(train_csv_path, vocab)
    val_dataset = AdditionDataset(val_csv_path, vocab)
    test_dataset = AdditionDataset(test_csv_path, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
