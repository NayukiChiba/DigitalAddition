# DigitalAddition

使用无注意力 Seq2Seq(Encoder-Decoder)模型实现 100 以内数字加减法.

## 项目简介

本项目构建一个基于 RNN 的序列到序列模型,**不使用注意力机制**,让模型直接学习将数学表达式(如 "35+27")映射到计算结果(如 "62").

- **输入**:`35+27` 或 `81-46`(字符级 token 序列)
- **输出**:`62` 或 `35`(字符级 token 序列,结果范围 0~199)
- **模型**:Encoder-Decoder(可选 SimpleRNN / LSTM / GRU),无 Attention
- **约束**:输入数字范围 0~99,运算符仅 `+` 和 `-`,结果非负

## 项目架构

```
DigitalAddition/
├── config/                          # 配置模块
│   ├── __init__.py
│   ├── paths.py                     # 路径常量
│   └── defaults.py                  # 默认超参数(dataclass)
├── src/
│   ├── __init__.py
│   ├── cli/                         # CLI 模块
│   │   ├── __init__.py
│   │   ├── parser.py                # 参数解析器(argparse)
│   │   └── menu.py                  # 子命令路由分发
│   ├── data/                        # 数据模块
│   │   ├── __init__.py
│   │   ├── generate.py              # 数据集生成(随机加减法算式)
│   │   ├── mapping.py               # 词表映射(char <-> index)
│   │   └── dataloader.py            # DataLoader 构建
│   ├── model/                       # 模型模块
│   │   ├── __init__.py              # 注册表 + buildModel()
│   │   ├── encoder.py               # Encoder(RNN/LSTM/GRU)
│   │   ├── decoder.py               # Decoder(RNN/LSTM/GRU,无 Attention)
│   │   └── seq2seq.py               # Seq2Seq 封装
│   ├── train/                       # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py               # 训练主循环(含 teacher forcing)
│   │   ├── optimizer.py             # 优化器构建
│   │   ├── scheduler.py             # 学习率调度器
│   │   ├── earlyStopping.py         # 早停机制
│   │   ├── checkpoint.py            # Checkpoint 管理
│   │   ├── logger.py                # 训练日志
│   │   └── utils.py                 # 工具函数
│   ├── evaluate/                    # 评估模块
│   │   ├── __init__.py
│   │   ├── evaluator.py             # 评估器
│   │   ├── metrics.py               # 评估指标(准确率、逐位匹配率)
│   │   └── visualize.py             # 可视化(训练曲线、错误分布)
│   └── inference/                   # 推理模块
│       ├── __init__.py
│       └── predictor.py             # 推理器(贪心解码)
├── datasets/                        # 数据集目录
│   ├── raw/                         # 原始生成数据(train/val/test CSV)
│   └── processed/                   # 预处理缓存(tokenized 序列)
├── outputs/                         # 输出目录
│   ├── checkpoints/                 # 模型权重
│   ├── logs/                        # 训练日志
│   └── figures/                     # 评估图表
├── notebooks/                       # 探索性分析
├── tests/                           # 单元测试
├── main.py                          # CLI 主入口
└── pyproject.toml                   # 项目元数据
```

## 快速开始

### 环境要求

- Python >= 3.11
- PyTorch >= 2.0(CUDA 可选)
- 其余依赖见 `pyproject.toml`

### 安装

```bash
# 克隆仓库
git clone https://github.com/NayukiChiba/DigitalAddition.git
cd DigitalAddition

# 创建虚拟环境并安装依赖
uv sync
uv sync --group dev
```

### 数据准备

```bash
# 生成训练/验证/测试数据集
python main.py data generate --train-size 50000 --val-size 5000 --test-size 5000
```

### 训练

```bash
# 使用 LSTM 训练
python main.py train --model lstm --epochs 50 --batch-size 128 --lr 0.001

# 使用 GRU 训练
python main.py train --model gru --epochs 50 --batch-size 128 --lr 0.001

# 使用 SimpleRNN 训练
python main.py train --model rnn --epochs 50 --batch-size 128 --lr 0.001
```

### 评估

```bash
python main.py eval --checkpoint outputs/checkpoints/best.pt
```

### 推理

```bash
# 单条推理
python main.py predict --checkpoint outputs/checkpoints/best.pt --expr "35+27"

# 批量推理
python main.py predict --checkpoint outputs/checkpoints/best.pt --file tests.txt
```

## 技术要点

### 数据生成

- 随机生成 a ∈ [0, 99], b ∈ [0, 99]
- 运算符随机抽取 `+` 或 `-`
- 若为减法且 a < b,交换 a, b 保证结果非负
- 输入序列最大长度 5(如 `99+99`),输出序列最大长度 3(最大结果 198)
- 特殊 token:`<SOS>`(解码开始)、`<EOS>`(解码结束)、`<PAD>`(填充)、`<UNK>`(未知)

### 模型架构

- **Encoder**:将输入字符序列编码为上下文向量(取最后时刻 hidden state)
- **Decoder**:从上下文向量自回归解码,Teacher Forcing 训练,贪心解码推理
- **无 Attention**:Decoder 仅依赖 Encoder 的最终 hidden state,强制模型学习更紧凑的表征

### 训练策略

- 损失函数:CrossEntropyLoss(忽略 `<PAD>`)
- Teacher Forcing 比例:可配置(默认 0.5 概率)
- 优化器:AdamW
- 学习率调度:ReduceLROnPlateau
- 早停:监控 val loss,patience 可配置
- 梯度裁剪

### 评估指标

- 完全匹配准确率(Exact Match):预测序列与真实序列完全一致的样本比例
- 逐位准确率(Character Accuracy):每个位置预测正确的字符占比
- 区分加减法的分项准确率

## 注意事项

- 无注意力意味着模型必须将整个输入表达式的语义压缩到一个固定维度的上下文向量中,对于较长的输入序列(如 `99+99` = 5 字符)信息压缩是可行的瓶颈测试
- 结果严格非负(减法保证 a >= b)
- 输入输出均以字符串形式处理,避免数值计算的捷径

