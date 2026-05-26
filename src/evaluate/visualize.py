"""
可视化模块

基于 matplotlib 提供训练相关的可视化功能：
- 训练历史曲线（损失 + 准确率双 Y 轴）
- 预测样本展示（输入 -> 真实输出 vs 预测输出）
- 错误分析（按表达式长度/运算符分类的准确率）

设计决策：
- 所有绘图函数均将图表保存到文件而非 plt.show()
  原因：训练通常在无图形界面的服务器上运行，plt.show() 会阻塞或无输出
- 双 Y 轴设计：左轴=损失（量级 0~N），右轴=准确率（量级 0~1）
  分开坐标轴避免损失曲线压扁准确率曲线
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 无图形界面环境（服务器/容器）使用非交互式后端
# 必须在 import pyplot 之后、任何绘图操作之前设置
matplotlib.use("Agg")

from config.paths import FIGURES_DIR


def setup_chinese_font():
    """
    配置中文字体支持

    按优先级尝试多个常见中文字体，避免因缺少字体导致中文显示为方块。
    仅在首次调用时执行，重复调用无副作用。
    """
    chinese_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> Path:
    """
    绘制训练历史曲线（双 Y 轴）

    左轴：train_loss + val_loss（损失越低越好）
    右轴：val_accuracy（准确率越高越好）

    双 Y 轴的必要性：
    损失通常在 0~N 范围，准确率在 0~1 范围。
    如果共用单轴，损失曲线会压扁准确率曲线使其几乎不可见。

    Args:
        history: 训练历史字典，至少包含 train_loss / val_loss / val_accuracy
        save_path: 图表保存路径，默认 FIGURES_DIR / training_history.png

    Returns:
        保存的文件路径
    """
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "training_history.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    fig, left_axis = plt.subplots(figsize=(10, 6))

    # 左 Y 轴：损失
    color_train = "#1f77b4"  # 蓝色
    color_val = "#ff7f0e"  # 橙色
    left_axis.set_xlabel("Epoch", fontsize=12)
    left_axis.set_ylabel("Loss", fontsize=12, color="black")

    if "train_loss" in history and len(history["train_loss"]) > 0:
        left_axis.plot(
            epochs,
            history["train_loss"],
            color=color_train,
            linewidth=1.5,
            marker="o",
            markersize=4,
            label="Train Loss",
        )
    if "val_loss" in history and len(history["val_loss"]) > 0:
        left_axis.plot(
            epochs,
            history["val_loss"],
            color=color_val,
            linewidth=1.5,
            marker="s",
            markersize=4,
            label="Val Loss",
        )

    left_axis.tick_params(axis="y")
    left_axis.grid(True, alpha=0.3)

    # 右 Y 轴：准确率
    right_axis = left_axis.twinx()
    right_axis.set_ylabel("Accuracy", fontsize=12, color="green")

    if "val_accuracy" in history and len(history["val_accuracy"]) > 0:
        right_axis.plot(
            epochs,
            history["val_accuracy"],
            color="green",
            linewidth=1.5,
            marker="^",
            markersize=4,
            label="Val Accuracy",
        )
    right_axis.set_ylim(0, 1.05)
    right_axis.tick_params(axis="y", labelcolor="green")

    # 合并图例
    lines_left, labels_left = left_axis.get_legend_handles_labels()
    lines_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper right",
        fontsize=10,
    )

    plt.title("Training History", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_prediction_samples(
    expressions: List[str],
    predictions: List[str],
    targets: List[str],
    num_samples: int = 20,
    save_path: Optional[Path] = None,
) -> Path:
    """
    绘制预测样本对比表格

    将输入表达式、真实答案、模型预测三列并排展示，
    错误预测以红色高亮，正确预测以默认颜色显示。

    随机抽取 num_samples 条样本展示，避免图表过长。

    Args:
        expressions: 输入表达式列表，如 ["35+27", "80-12"]
        predictions: 模型预测结果列表，如 ["62", "68"]
        targets:     真实答案列表，如 ["62", "68"]
        num_samples: 展示样本数
        save_path:   图表保存路径，默认 FIGURES_DIR / prediction_samples.png

    Returns:
        保存的文件路径
    """
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "prediction_samples.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 随机抽样
    total = len(expressions)
    if total > num_samples:
        indices = np.random.choice(total, num_samples, replace=False)
    else:
        indices = range(total)
        num_samples = total

    # 逐行收集表格数据
    rows: List[List[str]] = []
    cell_colors: List[List[str]] = []

    for idx in indices:
        is_correct = predictions[idx] == targets[idx]
        rows.append([expressions[idx], targets[idx], predictions[idx]])
        if is_correct:
            cell_colors.append(["white", "white", "white"])
        else:
            # 错误行为浅红色背景
            cell_colors.append(["#ffcccc", "#ffcccc", "#ffcccc"])

    fig, ax = plt.subplots(figsize=(12, num_samples * 0.4 + 1))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Expression", "Target", "Prediction"],
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.3)

    # 表头样式
    for col_idx in range(3):
        header_cell = table[0, col_idx]
        header_cell.set_facecolor("#4472C4")
        header_cell.set_text_props(color="white", fontweight="bold")

    plt.title("Prediction Samples (Red = Error)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_error_by_length(
    expressions: List[str],
    predictions: List[str],
    targets: List[str],
    save_path: Optional[Path] = None,
) -> Path:
    """
    按表达式长度分组绘制准确率柱状图

    目的：分析模型在不同长度输入上的表现差异。
    RNN 对长序列的建模能力通常弱于短序列，
    按长度分组的准确率可以直观展示这一趋势。

    Args:
        expressions: 输入表达式列表
        predictions: 模型预测结果列表
        targets:     真实答案列表
        save_path:   图表保存路径，默认 FIGURES_DIR / error_by_length.png

    Returns:
        保存的文件路径
    """
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "error_by_length.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 按表达式长度分组统计
    length_stats: Dict[int, Dict[str, int]] = {}
    for expression, prediction, target in zip(expressions, predictions, targets):
        expression_length = len(expression)
        if expression_length not in length_stats:
            length_stats[expression_length] = {"correct": 0, "total": 0}
        length_stats[expression_length]["total"] += 1
        if prediction == target:
            length_stats[expression_length]["correct"] += 1

    sorted_lengths = sorted(length_stats.keys())
    accuracies = [
        length_stats[length]["correct"] / length_stats[length]["total"]
        for length in sorted_lengths
    ]
    totals = [length_stats[length]["total"] for length in sorted_lengths]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        [str(length) for length in sorted_lengths],
        accuracies,
        color="#4472C4",
        alpha=0.85,
    )
    ax.set_xlabel("Expression Length (characters)", fontsize=12)
    ax.set_ylabel("Exact Match Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # 柱顶标注样本数
    for bar, total, accuracy in zip(bars, totals, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={total}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="gray",
        )

    plt.title(
        "Exact Match Accuracy by Expression Length", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_error_by_operator(
    expressions: List[str],
    predictions: List[str],
    targets: List[str],
    save_path: Optional[Path] = None,
) -> Path:
    """
    按运算符分组绘制准确率柱状图

    目的：分析模型在加法和减法上的表现差异。
    减法通常比加法更难（涉及借位），
    分组统计可以暴露模型是否存在对某种运算的偏好。

    Args:
        expressions: 输入表达式列表
        predictions: 模型预测结果列表
        targets:     真实答案列表
        save_path:   图表保存路径，默认 FIGURES_DIR / error_by_operator.png

    Returns:
        保存的文件路径
    """
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "error_by_operator.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 按运算符分组统计
    operator_stats: Dict[str, Dict[str, int]] = {}
    for expression, prediction, target in zip(expressions, predictions, targets):
        # 确定运算符类型
        if "+" in expression:
            operator_type = "+ (Addition)"
        elif "-" in expression:
            operator_type = "- (Subtraction)"
        else:
            continue

        if operator_type not in operator_stats:
            operator_stats[operator_type] = {"correct": 0, "total": 0}
        operator_stats[operator_type]["total"] += 1
        if prediction == target:
            operator_stats[operator_type]["correct"] += 1

    operator_names = list(operator_stats.keys())
    accuracies = [
        operator_stats[operator_name]["correct"]
        / operator_stats[operator_name]["total"]
        for operator_name in operator_names
    ]
    totals = [
        operator_stats[operator_name]["total"] for operator_name in operator_names
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4472C4", "#ED7D31"]
    bars = ax.bar(
        operator_names, accuracies, color=colors[: len(operator_names)], alpha=0.85
    )
    ax.set_ylabel("Exact Match Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar, total, accuracy in zip(bars, totals, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={total}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="gray",
        )

    plt.title("Exact Match Accuracy by Operator Type", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path


def plot_metrics_report(
    metrics: dict,
    save_path: Optional[Path] = None,
) -> Path:
    """
    绘制评估指标汇总图

    以水平柱状图 + 数值标注展示 Exact Match 和 Token Accuracy。

    Args:
        metrics: compute_metrics() 返回的指标字典
        save_path: 图表保存路径，默认 FIGURES_DIR / metrics_report.png

    Returns:
        保存的文件路径
    """
    setup_chinese_font()

    if save_path is None:
        save_path = FIGURES_DIR / "metrics_report.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 3))

    metric_names = ["Exact Match", "Token Accuracy"]
    metric_values = [
        metrics.get("exact_match", 0.0) * 100,
        metrics.get("token_accuracy", 0.0) * 100,
    ]
    bar_colors = ["#4472C4", "#ED7D31"]

    bars = ax.barh(
        metric_names, metric_values, color=bar_colors, alpha=0.85, height=0.5
    )
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    # 柱端标注具体数值
    for bar, value in zip(bars, metric_values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.title("Evaluation Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return save_path
