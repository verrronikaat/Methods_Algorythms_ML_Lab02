"""Метод потенциальных функций."""
import numpy as np
from .utils import get_distance_func
from .parzen import KERNELS


class PotentialClassifier:
    """Метод потенциальных функций с фиксированным h."""

    def __init__(self, h=1.0, kernel="gaussian", metric="euclidean"):
        self.h = float(h)
        self.kernel = kernel
        self.metric = metric
        self.X = None
        self.y = None
        self._dist = get_distance_func(metric)
        self._K = KERNELS[kernel]

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def _predict_one(self, x):
        dists = self._dist(self.X, x)
        r = dists / self.h
        w = self._K(r)
        score = np.zeros(2, dtype=float)
        for lab, wi in zip(self.y, w):
            score[int(lab)] += wi
        return int(np.argmax(score))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])