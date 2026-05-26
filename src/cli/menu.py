"""
CLI 模块 - 交互式菜单 + 命令行分发

提供两种使用方式：
1. 交互式菜单（show_menu）：无参数启动时进入，逐步引导输入
2. 命令行分发（dispatch）：带参数启动时直接执行，适合脚本化

核心逻辑抽取为 _do_train / _do_eval / _do_predict 三个内部函数，
交互菜单和 CLI 模式共享同一套实现。
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from config.defaults import DataParams, ModelParams, TrainingParams
from config.paths import (
    BEST_MODEL_PATH,
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    RAW_VAL_PATH,
)
from src.data.dataloader import create_data_loaders
from src.data.generate import generate_datasets
from src.data.mapping import VocabMapping, build_vocab
from src.evaluate.evaluator import Evaluator
from src.evaluate.visualize import plot_training_history
from src.inference.predictor import Predictor
from src.model import build_model
from src.train.checkpoint import load_checkpoint
from src.train.early_stopping import EarlyStopping
from src.train.optimizer import build_optimizer
from src.train.scheduler import build_scheduler
from src.train.trainer import Trainer
from src.train.utils import get_device, set_seed

# ======================================================================
# 交互式菜单辅助
# ======================================================================


def _input_with_default(prompt: str, default: str) -> str:
    """读取用户输入，回车使用默认值"""
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else str(default)


def _input_path(prompt: str) -> Path | None:
    """读取文件路径，回车跳过"""
    value = input(f"{prompt} (回车跳过): ").strip()
    return Path(value) if value else None


# ======================================================================
# 核心逻辑（交互菜单和 CLI 共用）
# ======================================================================


def _do_train(
    rnn_type: str = ModelParams.RNN_TYPE,
    hidden_dim: int = ModelParams.HIDDEN_DIM,
    dropout: float = ModelParams.DROPOUT,
    teacher_forcing_ratio: float = ModelParams.TEACHER_FORCING_RATIO,
    epochs: int = TrainingParams.EPOCHS,
    batch_size: int = DataParams.BATCH_SIZE,
    learning_rate: float = TrainingParams.LEARNING_RATE,
    optimizer_type: str = TrainingParams.OPTIMIZER,
    weight_decay: float = TrainingParams.WEIGHT_DECAY,
    grad_clip: float = TrainingParams.GRAD_CLIP,
    scheduler_type: str = TrainingParams.LR_SCHEDULER,
    lr_step_size: int = TrainingParams.LR_STEP_SIZE,
    lr_gamma: float = TrainingParams.LR_GAMMA,
    early_stop_patience: int = TrainingParams.EARLY_STOP_PATIENCE,
    early_stop_min_delta: float = TrainingParams.EARLY_STOP_MIN_DELTA,
    skip_generate: bool = False,
    resume_path: Path | None = None,
) -> None:
    """
    执行模型训练（核心逻辑）

    所有参数均有来自 config 的默认值，
    交互菜单通过用户输入覆盖，CLI 通过命令行参数覆盖。
    """
    set_seed()
    device = get_device()
    print(f"设备: {device}")

    # 数据准备
    if not skip_generate:
        print("生成数据集...")
        generate_datasets(
            train_size=DataParams.TRAIN_SIZE,
            val_size=DataParams.VAL_SIZE,
            test_size=DataParams.TEST_SIZE,
        )
    else:
        for csv_path in [RAW_TRAIN_PATH, RAW_VAL_PATH, RAW_TEST_PATH]:
            if not csv_path.exists():
                print(f"错误: 数据文件不存在: {csv_path}")
                return

    vocab = build_vocab()
    print(f"词表大小: {vocab.vocabulary_size}")

    train_loader, valid_loader, test_loader = create_data_loaders(
        vocab, batch_size=batch_size
    )

    # 模型构建
    model = build_model(
        vocab_size=vocab.vocabulary_size,
        pad_index=vocab.pad_index,
        sos_index=vocab.sos_index,
        eos_index=vocab.eos_index,
        rnn_type=rnn_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")

    # 训练组件
    optimizer = build_optimizer(
        model,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        step_size=lr_step_size,
        gamma=lr_gamma,
        epochs=epochs,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_index)
    early_stopping = EarlyStopping(
        patience=early_stop_patience,
        min_delta=early_stop_min_delta,
    )

    # 断点续训
    if resume_path is not None:
        if not resume_path.exists():
            print(f"错误: checkpoint 文件不存在: {resume_path}")
            return
        print(f"从 checkpoint 恢复: {resume_path}")
        checkpoint = load_checkpoint(
            resume_path, model, optimizer, scheduler, device=str(device)
        )
        if "early_stopping_state" in checkpoint:
            early_stopping.load_state_dict(checkpoint["early_stopping_state"])
        print(f"恢复到 epoch {checkpoint['epoch'] + 1}")

    # 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        early_stopping=early_stopping,
        vocab=vocab,
        device=device,
        epochs=epochs,
        grad_clip=grad_clip,
        teacher_forcing_ratio=teacher_forcing_ratio,
    )

    history = trainer.train()

    # 训练后评估
    plot_training_history(history)

    print("\n开始测试集评估...")
    best_checkpoint = load_checkpoint(BEST_MODEL_PATH, model, device=str(device))
    model.load_state_dict(best_checkpoint["model_state_dict"])

    evaluator = Evaluator(model, test_loader, vocab, device)
    evaluator.evaluate(history=history)


def _do_eval(
    checkpoint_path: Path,
    batch_size: int = DataParams.BATCH_SIZE,
) -> None:
    """执行模型评估（核心逻辑）"""
    device = get_device()
    print(f"设备: {device}")

    if not checkpoint_path.exists():
        print(f"错误: checkpoint 文件不存在: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab = VocabMapping.from_dict(checkpoint["vocab"])
    print(f"词表大小: {vocab.vocabulary_size}")

    if not RAW_TEST_PATH.exists():
        print("测试数据不存在，正在生成...")
        generate_datasets()

    _, _, test_loader = create_data_loaders(vocab, batch_size=batch_size)

    model = build_model(
        vocab_size=vocab.vocabulary_size,
        pad_index=vocab.pad_index,
        sos_index=vocab.sos_index,
        eos_index=vocab.eos_index,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    evaluator = Evaluator(model, test_loader, vocab, device)
    evaluator.evaluate()


def _do_predict(
    checkpoint_path: Path,
    expression: str | None = None,
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    """执行推理预测（核心逻辑）"""
    device = get_device()

    if not checkpoint_path.exists():
        print(f"错误: checkpoint 文件不存在: {checkpoint_path}")
        return

    predictor = Predictor.from_checkpoint(checkpoint_path, device)

    if expression is not None:
        result = predictor.predict(expression)
        print(f"表达式: {expression}")
        print(f"结果:   {result}")
    elif input_path is not None and output_path is not None:
        predictor.predict_file(input_path, output_path)
    else:
        print("请指定 --expression 或 --input/--output")


# ======================================================================
# 交互式菜单
# ======================================================================


def menu_train() -> None:
    """交互式训练向导：逐步引导用户输入参数"""
    print()
    print("=" * 50)
    print("  训练模型")
    print("=" * 50)
    print()

    # 模型结构
    print("\n[模型结构]")
    rnn_type = _input_with_default("  RNN 类型 (LSTM/RNN/GRU)", ModelParams.RNN_TYPE)
    hidden_dim = int(_input_with_default("  隐藏层维度", str(ModelParams.HIDDEN_DIM)))
    dropout = float(_input_with_default("  Dropout 概率", str(ModelParams.DROPOUT)))
    teacher_forcing_ratio = float(
        _input_with_default(
            "  Teacher Forcing 概率", str(ModelParams.TEACHER_FORCING_RATIO)
        )
    )

    # 训练超参数
    print("\n[训练超参数]")
    epochs = int(_input_with_default("  训练轮数", str(TrainingParams.EPOCHS)))
    batch_size = int(_input_with_default("  批大小", str(DataParams.BATCH_SIZE)))
    learning_rate = float(
        _input_with_default("  学习率", str(TrainingParams.LEARNING_RATE))
    )
    optimizer_type = _input_with_default(
        "  优化器 (Adam/SGD/AdamW)", TrainingParams.OPTIMIZER
    )
    weight_decay = float(
        _input_with_default("  权重衰减系数", str(TrainingParams.WEIGHT_DECAY))
    )
    grad_clip = float(
        _input_with_default("  梯度裁剪阈值", str(TrainingParams.GRAD_CLIP))
    )

    # 学习率调度
    print("\n[学习率调度]")
    scheduler_type = _input_with_default(
        "  调度器 (StepLR/CosineAnnealingLR/ReduceLROnPlateau)",
        TrainingParams.LR_SCHEDULER,
    )
    lr_step_size = int(
        _input_with_default("  StepLR 衰减周期", str(TrainingParams.LR_STEP_SIZE))
    )
    lr_gamma = float(
        _input_with_default("  StepLR 衰减因子", str(TrainingParams.LR_GAMMA))
    )

    # 早停
    print("\n[早停]")
    early_stop_patience = int(
        _input_with_default("  容忍轮数", str(TrainingParams.EARLY_STOP_PATIENCE))
    )
    early_stop_min_delta = float(
        _input_with_default("  最小改善阈值", str(TrainingParams.EARLY_STOP_MIN_DELTA))
    )

    # 数据
    print("\n[数据]")
    skip_generate = input("  跳过数据生成？(y/N): ").strip().lower() == "y"

    # 断点续训
    resume_path = _input_path("  恢复训练的 checkpoint 路径")

    # 调用核心逻辑
    _do_train(
        rnn_type=rnn_type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        teacher_forcing_ratio=teacher_forcing_ratio,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        scheduler_type=scheduler_type,
        lr_step_size=lr_step_size,
        lr_gamma=lr_gamma,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        skip_generate=skip_generate,
        resume_path=resume_path,
    )


def menu_eval() -> None:
    """交互式评估向导"""
    print()
    print("=" * 50)
    print("  评估模型")
    print("=" * 50)
    print()

    checkpoint_input = input("  checkpoint 文件路径: ").strip()
    if not checkpoint_input:
        print("错误: 必须指定 checkpoint 路径")
        return
    checkpoint_path = Path(checkpoint_input)

    batch_size = int(_input_with_default("  批大小", str(DataParams.BATCH_SIZE)))

    _do_eval(checkpoint_path, batch_size)


def menu_predict() -> None:
    """交互式推理向导"""
    print()
    print("=" * 50)
    print("  推理预测")
    print("=" * 50)
    print()

    checkpoint_input = input("  checkpoint 文件路径: ").strip()
    if not checkpoint_input:
        print("错误: 必须指定 checkpoint 路径")
        return
    checkpoint_path = Path(checkpoint_input)

    print()
    print("  推理模式:")
    print("    1. 单条表达式计算")
    print("    2. CSV 文件批量推理")
    print()
    mode = input("  请选择 (1/2): ").strip()

    if mode == "1":
        expression = input("  输入表达式 (如 35+27): ").strip()
        if not expression:
            print("错误: 表达式不能为空")
            return
        _do_predict(checkpoint_path, expression=expression)

    elif mode == "2":
        input_path_str = input("  输入 CSV 文件路径: ").strip()
        output_path_str = input("  输出 CSV 文件路径: ").strip()
        if not input_path_str or not output_path_str:
            print("错误: 路径不能为空")
            return
        _do_predict(
            checkpoint_path,
            input_path=Path(input_path_str),
            output_path=Path(output_path_str),
        )

    else:
        print("无效选择")


def show_menu() -> None:
    """显示主菜单循环"""
    while True:
        print()
        print("=" * 50)
        print("  DigitalAddition - Seq2Seq 加减法计算")
        print("=" * 50)
        print("  1. 训练模型")
        print("  2. 评估模型")
        print("  3. 推理预测")
        print("  0. 退出")
        print("-" * 50)

        choice = input("  请选择 (0-3): ").strip()

        if choice == "1":
            menu_train()
        elif choice == "2":
            menu_eval()
        elif choice == "3":
            menu_predict()
        elif choice in ("0", "q", "Q"):
            print("再见！")
            break
        else:
            print("无效选择，请重新输入")


# ======================================================================
# CLI 命令行分发
# ======================================================================


def dispatch(args: argparse.Namespace) -> None:
    """
    根据解析后的命令行参数分发到对应逻辑

    与交互菜单共享 _do_train / _do_eval / _do_predict 核心函数，
    直接传入命令行参数，无交互提示。
    """
    if args.command == "train":
        _do_train(
            rnn_type=args.rnn_type,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            scheduler_type=args.lr_scheduler,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            skip_generate=args.skip_generate,
            resume_path=Path(args.resume) if args.resume else None,
        )

    elif args.command == "eval":
        _do_eval(
            checkpoint_path=Path(args.checkpoint),
            batch_size=args.batch_size,
        )

    elif args.command == "predict":
        expression = args.expression
        input_path = Path(args.input) if args.input else None
        output_path = Path(args.output) if args.output else None
        _do_predict(
            checkpoint_path=Path(args.checkpoint),
            expression=expression,
            input_path=input_path,
            output_path=output_path,
        )

    else:
        print(f"未知子命令: {args.command}")
        print("使用 --help 查看帮助")
