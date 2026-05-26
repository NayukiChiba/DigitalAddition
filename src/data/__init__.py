"""
数据模块

统一导出数据处理相关的所有组件：
- generate_datasets: 数据集生成入口
- VocabMapping:      词表映射管理
- AdditionDataset:   PyTorch Dataset 封装
- create_data_loaders: DataLoader 工厂函数
"""

from src.data.dataloader import AdditionDataset, create_data_loaders
from src.data.generate import generate_datasets
from src.data.mapping import VocabMapping, build_vocab

__all__ = [
    "generate_datasets",
    "VocabMapping",
    "build_vocab",
    "AdditionDataset",
    "create_data_loaders",
]
