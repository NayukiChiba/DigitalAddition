"""
日志记录模块

提供 Logger 类,同时支持 CSV 文本日志和 TensorBoard 两种记录方式.

日志产出(每个实验一个子目录,命名 = 时间戳):
    outputs/logs/20250101_120000/
    ├── metrics.csv     — epoch 级别指标表格,可用 pandas 读取分析
    └── train.log       — 文本日志,记录启动/结束/异常等关键事件

    outputs/tensorboard/ — TensorBoard 事件文件(可选,需安装 tensorboard)

设计决策:
- CSV 而非 JSON:CSV 更易于在 Excel 中打开对比多次实验的指标
- 每 epoch 立刻 flush:防止程序崩溃丢失所有日志
- TensorBoard 可选:不是所有环境都安装 tensorboard,导入失败时静默跳过
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from config.paths import LOGS_DIR, TENSORBOARD_DIR


class Logger:
    """
    训练日志记录器

    同时将指标写入 CSV 文件和 TensorBoard(若可用).

    使用方式:
        logger = Logger()
        logger.start()

        for epoch in range(epochs):
            logger.log_epoch(
                metrics={"train_loss": 0.35, "val_loss": 0.42, "val_acc": 0.85},
                epoch=epoch,
            )

        logger.close()
    """

    def __init__(
        self,
        log_dir: Path = LOGS_DIR,
        tensorboard_dir: Path = TENSORBOARD_DIR,
        experiment_name: Optional[str] = None,
    ):
        """
        Args:
            log_dir:          CSV/文本日志根目录,默认 LOGS_DIR
            tensorboard_dir:  TensorBoard 事件目录,默认 TENSORBOARD_DIR
            experiment_name:  实验名称(子目录名),默认用当前时间戳
        """
        self.log_dir = log_dir
        self.tensorboard_dir = tensorboard_dir
        # 每次训练创建独立目录,避免覆盖历史日志
        self.experiment_name = (
            experiment_name
            if experiment_name is not None
            else datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        # 实验输出目录: logs/20250101_120000/
        self.run_dir = self.log_dir / self.experiment_name

        # 持久化资源,start() 时初始化,close() 时释放
        self.csv_writer: Optional[csv.DictWriter] = None
        self.csv_file = None  # 文件句柄
        self.writer = None  # TensorBoard SummaryWriter
        self.is_active: bool = False

    def start(self) -> None:
        """
        启动日志记录

        创建实验子目录,打开 CSV 文件,尝试初始化 TensorBoard.
        CSV 表头在第一次 log_epoch 时根据传入的 metrics key 动态生成,
        因此 start() 时不需要知道指标名称.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 打开 CSV 文件(表头延迟到首次 log_epoch 时写入)
        csv_path = self.run_dir / "metrics.csv"
        self.csv_file = csv_path.open("w", encoding="utf-8", newline="")

        # TensorBoard(可选,导入失败时跳过)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        except ImportError:
            self.writer = None

        self.is_active = True
        self._write_event_log("日志已启动")

    def log_epoch(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        记录单个 epoch 的指标

        首次调用时自动根据 metrics 的 key 写入 CSV 表头.
        后续调用必须保持 key 一致,否则 CSV 列会错位.

        Args:
            metrics: 指标字典,如 {"train_loss": 0.35, "val_loss": 0.42}
            epoch:   epoch 编号(0-based)
        """
        if not self.is_active:
            return

        # 首次调用:根据 metrics 的 key 初始化 CSV 表头
        if self.csv_writer is None:
            fieldnames = ["epoch"] + list(metrics.keys())
            self.csv_writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
            )
            self.csv_writer.writeheader()

        # 写入一行 CSV 数据
        row = {"epoch": epoch, **metrics}
        self.csv_writer.writerow(row)
        # 立即刷盘:如果程序崩溃,至少不会丢失整份日志
        self.csv_file.flush()

        # 写入 TensorBoard(每个指标一条曲线)
        if self.writer is not None:
            for tag, value in metrics.items():
                self.writer.add_scalar(tag, value, epoch)

    def log_message(self, message: str) -> None:
        """
        记录一条文本事件消息

        用于记录训练开始/结束、异常等重要事件,
        写入 run_dir 下的 train.log 文件.
        """
        self._write_event_log(message)

    def close(self) -> None:
        """
        关闭日志记录

        释放 TensorBoard writer 和 CSV 文件句柄.
        未正确 close 可能导致 CSV 最后几行未刷盘.
        """
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        self.is_active = False

    def _write_event_log(self, message: str) -> None:
        """写入带时间戳的文本日志行"""
        log_path = self.run_dir / "train.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
