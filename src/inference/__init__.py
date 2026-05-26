"""
推理模块

统一导出的组件：
- Predictor — Seq2Seq 算术推理器
"""

from src.inference.predictor import Predictor

__all__ = [
    "Predictor",
]
