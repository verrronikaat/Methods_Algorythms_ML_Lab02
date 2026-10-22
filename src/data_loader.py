"""Загрузка датасетов GoT (локально или из GitHub)."""
import pandas as pd
from . import config as cfg


def load_data(force_download=False):
    """Возвращает (train, test). Скачивает, если локально нет."""
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = cfg.DATA_DIR / "game_of_thrones_train.csv"
    test_path = cfg.DATA_DIR / "game_of_thrones_test.csv"

    if force_download or not train_path.exists():
        train = pd.read_csv(cfg.DATA_TRAIN_URL)
        train.to_csv(train_path, index=False)
    else:
        train = pd.read_csv(train_path)

    if force_download or not test_path.exists():
        test = pd.read_csv(cfg.DATA_TEST_URL)
        test.to_csv(test_path, index=False)
    else:
        test = pd.read_csv(test_path)

    return train, test