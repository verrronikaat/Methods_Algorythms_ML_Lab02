"""Параметры варианта 17."""
from pathlib import Path

# ------- Параметры варианта -------
SEED = 17
K_MAX = 14
H_MAX = 1.4
METRIC = "manhattan"                # euclidean / manhattan / minkowski
WEIGHTS = "uniform"                 # uniform / distance / linear / exponential
KERNEL = "rectangular"              # rectangular / gaussian / triangular / epanechnikov / quartic

# Popularity
POPULARITY_MODE = "qcut"            # qcut / log / raw
POPULARITY_Q = 10                   # число квантильных бинов

# Culture
CULTURE_MODE = "rare_merge"         # full / none / rare_merge
RARE_THRESHOLD = 5                  # порог для объединения редких

# House
HOUSE_MODE = "onehot"               # label / onehot / topk
HOUSE_TOPK = 10

# ANN (Annoy)
N_TREES = 11
N_NEIGHBORS_ANN = 9
QUERY_CHARACTER = "Petyr Baelish"

# ------- Пути -------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
SUBMISSION_PATH = ROOT / "submission.csv"

DATA_TRAIN_URL = (
    "https://raw.githubusercontent.com/EvADarya/"
    "Data_processing-DLS-/refs/heads/main/game_of_thrones_train.csv"
)
DATA_TEST_URL = (
    "https://raw.githubusercontent.com/EvADarya/"
    "Data_processing-DLS-/refs/heads/main/game_of_thrones_test.csv"
)

# Разбиение
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15