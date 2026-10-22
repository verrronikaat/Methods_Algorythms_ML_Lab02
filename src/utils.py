"""Утилиты: расстояния и метрики."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def minkowski_distance(X, x, p=2):
    """Обобщённая метрика Минковского: d(X, x) для всех строк X."""
    X = np.asarray(X, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.power(np.sum(np.abs(X - x) ** p, axis=1), 1.0 / p)


def euclidean_distance(X, x):
    return minkowski_distance(X, x, p=2)


def manhattan_distance(X, x):
    return minkowski_distance(X, x, p=1)


def get_distance_func(metric):
    """Возвращает функцию расстояния по имени метрики."""
    metric = metric.lower()
    if metric == "euclidean":
        return euclidean_distance
    if metric == "manhattan":
        return manhattan_distance
    if metric.startswith("minkowski"):
        return lambda X, x: minkowski_distance(X, x, p=3)
    raise ValueError(f"Unknown metric: {metric}")


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def f1_binary(y_true, y_pred):
    return f1_score(y_true, y_pred, average="binary", zero_division=0)


def print_metrics(name, y_true, y_pred):
    acc = accuracy(y_true, y_pred)
    f1 = f1_binary(y_true, y_pred)
    print(f"  {name:35s} acc={acc:.4f}  f1={f1:.4f}")
    return acc, f1