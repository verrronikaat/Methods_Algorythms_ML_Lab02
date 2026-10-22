"""Собственная реализация kNN и LOO-подбор k."""
import numpy as np
from .utils import get_distance_func


class KNNClassifier:
    """kNN с метрикой Минковского и различными весами."""

    def __init__(self, k=5, metric="euclidean", weights="uniform", p=2):
        self.k = int(k)
        self.metric = metric
        self.weights = weights
        self.p = p
        self.X = None
        self.y = None
        self._dist = get_distance_func(metric)

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def _weights(self, dists):
        k = len(dists)
        if self.weights == "uniform":
            return np.ones(k)
        if self.weights == "distance":
            return 1.0 / (dists + 1e-12)
        if self.weights == "linear":
            return (k + 1 - np.arange(1, k + 1)) / k
        if self.weights == "exponential":
            return 0.5 ** np.arange(k)
        raise ValueError(f"Unknown weights: {self.weights}")

    def _predict_one(self, x):
        dists = self._dist(self.X, x)
        idx = np.argsort(dists)[: self.k]
        w = self._weights(dists[idx])
        labels = self.y[idx]
        score = np.zeros(2, dtype=float)
        for lab, wi in zip(labels, w):
            score[int(lab)] += wi
        return int(np.argmax(score))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])


def loo_knn(X, y, k_max, metric="euclidean", weights="uniform"):
    """LOO-оценка ошибок для k = 1..k_max."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(X)
    dist = get_distance_func(metric)

    # матрица расстояний
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = dist(X, X[i])

    errors = {}
    for k in range(1, k_max + 1):
        wrong = 0
        for i in range(n):
            order = np.argsort(D[i])
            order = order[order != i][:k]
            d = D[i][order]
            if weights == "uniform":
                w = np.ones(len(order))
            elif weights == "distance":
                w = 1.0 / (d + 1e-12)
            elif weights == "linear":
                w = (len(order) + 1 - np.arange(1, len(order) + 1)) / len(order)
            elif weights == "exponential":
                w = 0.5 ** np.arange(len(order))
            else:
                raise ValueError(weights)
            score = np.zeros(2)
            for j, lab in enumerate(y[order]):
                score[int(lab)] += w[j]
            if int(np.argmax(score)) != y[i]:
                wrong += 1
        errors[k] = wrong / n
    best_k = min(errors, key=errors.get)
    return errors, best_k