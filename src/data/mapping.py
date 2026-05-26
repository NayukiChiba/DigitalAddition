"""
词表映射模块

提供 VocabMapping 类,封装字符级 token 与索引的双向映射,
支持序列化(JSON)和反序列化,便于 checkpoint 复用.

特殊 token:
- <PAD> = 0: 填充标记
- <UNK> = 1: 未知字符标记
- <SOS> = 2: 序列起始标记
- <EOS> = 3: 序列终止标记

字符集:0-9 数字、+、- 运算符
词表大小:10 数字 + 2 运算符 + 4 特殊 = 16
"""

import json
from pathlib import Path
from typing import Dict, List

DIGIT_CHARS = [str(i) for i in range(10)]
OPERATOR_CHARS = ["+", "-"]
ALL_CHARS = DIGIT_CHARS + OPERATOR_CHARS


class VocabMapping:
    """
    词表映射类

    封装 word_to_index 和 index_to_word 两个方向的映射字典,
    提供词表大小和特殊 token 索引的便捷属性.

    Attributes:
        word_to_index: 字符 → 索引 的映射字典
        index_to_word: 索引 → 字符 的映射字典
    """

    def __init__(self, word_to_index: Dict[str, int], index_to_word: Dict[int, str]):
        """
        初始化词表映射

        Args:
            word_to_index: 字符到索引的映射字典
            index_to_word: 索引到字符的映射字典
        """
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

    @property
    def vocabulary_size(self) -> int:
        """词表大小(含所有特殊 token)"""
        return len(self.word_to_index)

    @property
    def pad_index(self) -> int:
        """<PAD> 的索引,固定为 0"""
        return 0

    @property
    def unk_index(self) -> int:
        """<UNK> 的索引,固定为 1"""
        return 1

    @property
    def sos_index(self) -> int:
        """<SOS> 的索引,固定为 2"""
        return 2

    @property
    def eos_index(self) -> int:
        """<EOS> 的索引,固定为 3"""
        return 3

    # ——— 编码 / 解码 ———

    def encode(self, expr: str, add_sos_eos: bool = False) -> List[int]:
        """
        将字符串编码为索引序列

        Args:
            expr: 输入字符串,如 "35+27" 或 "62"
            add_sos_eos: 是否添加 <SOS>/<EOS> 首尾标记

        Returns:
            索引列表
        """
        indices = [self.sos_index] if add_sos_eos else []
        for ch in expr:
            indices.append(self.word_to_index.get(ch, self.unk_index))
        if add_sos_eos:
            indices.append(self.eos_index)
        return indices

    def decode(self, indices: List[int], strip_special: bool = True) -> str:
        """
        将索引序列解码为字符串

        Args:
            indices: 索引列表
            strip_special: 是否移除特殊 token 及 EOS 之后的内容

        Returns:
            解码后的字符串
        """
        skip = {self.pad_index, self.sos_index, self.eos_index}
        chars = []
        for idx in indices:
            if idx == self.eos_index:
                break
            if strip_special and idx in skip:
                continue
            chars.append(self.index_to_word.get(idx, "<UNK>"))
        return "".join(chars)

    # ——— 序列化 ———

    def to_dict(self) -> Dict[str, Dict]:
        """
        将词表序列化为可 JSON 序列化的字典

        Returns:
            包含 word_to_index 和 index_to_word 的字典
        """
        return {
            "word_to_index": self.word_to_index,
            "index_to_word": {str(k): v for k, v in self.index_to_word.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Dict]) -> "VocabMapping":
        """
        从序列化字典恢复 VocabMapping 实例

        Args:
            data: to_dict() 方法输出的字典

        Returns:
            重建的 VocabMapping 实例
        """
        word_to_index = data["word_to_index"]
        index_to_word = {int(k): v for k, v in data["index_to_word"].items()}
        return cls(word_to_index, index_to_word)

    def save(self, filepath: Path) -> None:
        """保存词表到 JSON 文件"""
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "VocabMapping":
        """从 JSON 文件加载词表"""
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return cls.from_dict(data)


# ——— 工厂函数 ———


def buildVocab() -> VocabMapping:
    """
    构建字符级词表(固定映射,不依赖数据)

    索引分配:
      <PAD>=0  <UNK>=1  <SOS>=2  <EOS>=3
      '0'=4  '1'=5  ...  '9'=13  '+'=14  '-'=15
    """
    word_to_index = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<SOS>": 2,
        "<EOS>": 3,
    }
    index_to_word = {
        0: "<PAD>",
        1: "<UNK>",
        2: "<SOS>",
        3: "<EOS>",
    }

    idx = 4
    for ch in ALL_CHARS:
        word_to_index[ch] = idx
        index_to_word[idx] = ch
        idx += 1

    return VocabMapping(word_to_index, index_to_word)
