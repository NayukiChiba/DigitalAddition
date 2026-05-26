"""
CLI 模块

提供两种启动方式：
- show_menu:   交互式控制台菜单（无参数启动）
- dispatch:    命令行参数分发（带子命令启动，用于脚本化）
- build_parser: 命令行参数解析器
"""

from src.cli.menu import dispatch, show_menu
from src.cli.parser import build_parser

__all__ = ["show_menu", "dispatch", "build_parser"]
