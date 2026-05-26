"""
数据集生成模块

随机生成 100 以内加减法算式及其结果,保存为 CSV 文件.

规则:
- 数字 a, b in [minNumber, maxNumber)
- 运算符随机抽取 + 或 -
- 减法时若 a < b 则交换 a, b,保证结果非负
- 输入序列:字符级表达式(如 "35+27")
- 输出序列:字符级结果(如 "62")

使用方法:
    python -m src.data.generate


"""

import csv
import random
from pathlib import Path

from config.defaults import DataParams, DefaultParams
from config.paths import (
    RAW_TEST_PATH,
    RAW_TRAIN_PATH,
    RAW_VAL_PATH,
)


def generate_one_data(
    minNumber: int = DataParams.MIN_NUMBER,
    maxNumber: int = DataParams.MAX_NUMBER,
    operators: list = DataParams.OPERATORS,
) -> tuple[str, str]:
    """
    生成一个随机的加减法算式及其结果
    Args:
        minNumber(int): 数字范围的最小值(包含)
        maxNumber(int): 数字范围的最大值(不包含)
        operators(list): 允许的运算符列表(如 ["+", "-"])

    Returns:
        input_seq(str): 字符级表达式(如 "35+27")
        output_seq(str): 字符级结果(如 "62")
    """
    a = random.randint(minNumber, maxNumber - 1)
    b = random.randint(minNumber, maxNumber - 1)
    op = random.choice(operators)

    if op == "+":
        result = a + b
    else:  # op == "-"
        # 减法要保证结果非负,如果 a < b 则交换 a, b
        if a < b:
            a, b = b, a
        result = a - b

    input_seq = f"{a}{op}{b}"
    output_seq = str(result)
    return input_seq, output_seq


def generate_samples(
    num_samples: int,
    min_number: int = DataParams.MIN_NUMBER,
    max_number: int = DataParams.MAX_NUMBER,
    operators: list = DataParams.OPERATORS,
) -> list[tuple[str, str]]:
    """
    生成指定数量的随机加减法算式及其结果
    Args:
        num_samples(int): 需要生成的样本数量
        min_number(int): 数字范围的最小值(包含)
        max_number(int): 数字范围的最大值(不包含)
        operators(list): 允许的运算符列表(如 ["+", "-"])

    Returns:
        samples(list of tuple): 包含 (input_seq, output_seq) 的列表
    """
    samples = []
    for _ in range(num_samples):
        samples.append(generate_one_data(min_number, max_number, operators))
    return samples


def save_to_csv(samples: list[tuple[str, str]], file_path: Path) -> None:
    """
    将生成的样本保存为 CSV 文件
    Args:
        samples(list of tuple): 包含 (input_seq, output_seq) 的列表
        file_path(Path): CSV 文件路径
    """
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["input", "output"])  # 写入表头
        writer.writerows(samples)  # 写入数据行
    print(f"Saved {len(samples)} samples to {file_path}")


def generate_datasets(
    train_size: int = DataParams.TRAIN_SIZE,
    val_size: int = DataParams.VAL_SIZE,
    test_size: int = DataParams.TEST_SIZE,
    seed: int = DefaultParams.RANDOM_SEED,
) -> None:
    """
    生成训练集、验证集和测试集并保存为 CSV 文件
    Args:
        outputs_dir(Path): 输出目录路径
        train_size(int): 训练集样本数量
        val_size(int): 验证集样本数量
        test_size(int): 测试集样本数量
        seed(int): 随机种子,保证每次生成相同的数据集


    """
    random.seed(seed)  # 设置随机种子

    # 生成训练集、验证集和测试集
    train_samples = generate_samples(train_size)
    val_samples = generate_samples(val_size)
    test_samples = generate_samples(test_size)

    # 保存为 CSV 文件
    save_to_csv(train_samples, RAW_TRAIN_PATH)
    save_to_csv(val_samples, RAW_VAL_PATH)
    save_to_csv(test_samples, RAW_TEST_PATH)


if __name__ == "__main__":
    generate_datasets()
