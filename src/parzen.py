"""Окно Парзена (aNN): фиксированная и переменная ширина."""
import numpy as np
from .utils import get_distance_func


def kernel_rectangular(r):
    return (np.abs(r) <= 1).astype(float)


def kernel_gaussian(r):
    return np.exp(-2 * r ** 2)


def kernel_triangular(r):
    return np.maximum(0.0, 1.0 - np.abs(r))


def kernel_epanechnikov(r):
    return np.where(np.abs(r) <= 1, 1.0 - r ** 2, 0.0)


def kernel_quartic(r):
    return np.where(np.abs(r) <= 1, (1.0 - r ** 2) ** 2, 0.0)


KERNELS = {
    "rectangular": kernel_rectangular,
    "gaussian": kernel_gaussian,
    "triangular": kernel_triangular,
    "epanechnikov": kernel_epanechnikov,
    "quartic": kernel_quartic,
}


class ParzenClassifier:
    """Окно Парзена (fixed / variable ширина)."""

    def __init__(self, h=1.0, kernel="rectangular", metric="euclidean",
                 mode="fixed", k=5):
        self.h = float(h)
        self.kernel = kernel
        self.metric = metric
        self.mode = mode
        self.k = int(k)
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
        if self.mode == "fixed":
            h = self.h
        else:
            sorted_d = np.sort(dists)
            h = sorted_d[min(self.k, len(sorted_d) - 1)] + 1e-12
        r = dists / h
        w = self._K(r)
        score = np.zeros(2, dtype=float)
        for lab, wi in zip(self.y, w):
            score[int(lab)] += wi
        return int(np.argmax(score))

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])


def loo_parzen(X, y, h_grid, kernel, metric="euclidean", mode="fixed", k=5):
    """LOO-подбор ширины h."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(X)
    dist = get_distance_func(metric)
    K = KERNELS[kernel]

    D = np.zeros((n, n))
    for i in range(n):
        D[i] = dist(X, X[i])

    errors = {}
    for h in h_grid:
        wrong = 0
        for i in range(n):
            d = np.delete(D[i], i)
            labs = np.delete(y, i)
            hh = h
            if mode == "variable":
                sorted_d = np.sort(d)
                hh = sorted_d[min(k, len(sorted_d) - 1)] + 1e-12
            r = d / hh
            w = K(r)
            score = np.zeros(2)
            for lab, wi in zip(labs, w):
                score[int(lab)] += wi
            if int(np.argmax(score)) != y[i]:
                wrong += 1
        errors[h] = wrong / n
    best_h = min(errors, key=errors.get)
    return errors, best_h