"""
评估模块

统一导出的组件：
- compute_metrics / compute_exact_match / compute_token_accuracy — 评估指标
- Evaluator — 测试集评估器
- plot_training_history / plot_prediction_samples / plot_error_by_length /
  plot_error_by_operator / plot_metrics_report — 可视化函数
"""

from src.evaluate.evaluator import Evaluator
from src.evaluate.metrics import (
    compute_exact_match,
    compute_metrics,
    compute_token_accuracy,
)
from src.evaluate.visualize import (
    plot_error_by_length,
    plot_error_by_operator,
    plot_metrics_report,
    plot_prediction_samples,
    plot_training_history,
)

__all__ = [
    "compute_metrics",
    "compute_exact_match",
    "compute_token_accuracy",
    "Evaluator",
    "plot_training_history",
    "plot_prediction_samples",
    "plot_error_by_length",
    "plot_error_by_operator",
    "plot_metrics_report",
]
