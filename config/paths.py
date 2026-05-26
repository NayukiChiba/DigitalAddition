from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


# 数据集位置
DATASETS_DIR = get_dir(ROOT / "datasets")
# 原始数据集和处理后数据集的位置
RAW_DATASETS_DIR = get_dir(DATASETS_DIR / "raw")
PROCESSED_DATASETS_DIR = get_dir(DATASETS_DIR / "processed")


# 产出位置
OUTPUTS_DIR = get_dir(ROOT / "outputs")

CHECKPOINTS_DIR = get_dir(OUTPUTS_DIR / "checkpoints")
LOGS_DIR = get_dir(OUTPUTS_DIR / "logs")
TENSORBOARD_DIR = get_dir(OUTPUTS_DIR / "tensorboard")
FIGURES_DIR = get_dir(OUTPUTS_DIR / "figures")

# 最佳模型保存和最后模型保存位置
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINTS_DIR / "last_model.pth"

# 原始数据集 CSV 文件
RAW_TRAIN_PATH = RAW_DATASETS_DIR / "train.csv"
RAW_VAL_PATH = RAW_DATASETS_DIR / "val.csv"
RAW_TEST_PATH = RAW_DATASETS_DIR / "test.csv"

# 预处理后的数据集文件
PROCESSED_TRAIN_PATH = PROCESSED_DATASETS_DIR / "train.pt"
PROCESSED_VAL_PATH = PROCESSED_DATASETS_DIR / "val.pt"
PROCESSED_TEST_PATH = PROCESSED_DATASETS_DIR / "test.pt"
PROCESSED_VOCAB_PATH = PROCESSED_DATASETS_DIR / "vocab.json"
