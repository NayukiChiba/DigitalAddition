from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path


# 数据集位置
DATASETS_DIR = get_dir(ROOT / "datasets")

# 产出位置
OUTPUTS_DIR = get_dir(ROOT / "outputs")

CHECKPOINTS_DIR = get_dir(OUTPUTS_DIR / "checkpoints")
LOGS_DIR = get_dir(OUTPUTS_DIR / "logs")
TENSORBOARD_DIR = get_dir(OUTPUTS_DIR / "tensorboard")
FIGURES_DIR = get_dir(OUTPUTS_DIR / "figures")

# 最佳模型保存和最后模型保存位置
BEST_MODEL_PATH = CHECKPOINTS_DIR / "best_model.pth"
LAST_MODEL_PATH = CHECKPOINTS_DIR / "last_model.pth"
