"""
DigitalAddition CLI 主入口

基于 Seq2Seq（无 Attention）的 100 以内加减法计算工具。

支持两种启动方式：

1. 交互式菜单（无参数）：
       python main.py

2. 命令行子命令（带参数）：
       python main.py train   [OPTIONS]      — 训练模型
       python main.py eval    --checkpoint <path> [OPTIONS]  — 评估模型
       python main.py predict --checkpoint <path> --expression "..."  — 推理计算

CLI 用法示例：
    python main.py train --rnn-type LSTM --epochs 50 --batch-size 128
    python main.py train --resume outputs/checkpoints/last_model.pth
    python main.py eval --checkpoint outputs/checkpoints/best_model.pth
    python main.py predict --checkpoint outputs/checkpoints/best_model.pth --expression "35+27"
    python main.py predict --checkpoint outputs/checkpoints/best_model.pth --input test.csv --output result.csv
"""

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，确保模块导入正确
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli.menu import dispatch, show_menu
from src.cli.parser import build_parser


def main() -> None:
    """主入口函数"""

    # 无参数 → 交互式菜单
    if len(sys.argv) == 1:
        show_menu()
        return

    # 有参数 → 命令行解析
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch(args)


if __name__ == "__main__":
    main()
