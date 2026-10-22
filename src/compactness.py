"""Профиль компактности и отбор эталонных объектов."""
import numpy as np
from .utils import get_distance_func


def _pairwise_distances(X, metric):
    """Считает матрицу расстояний один раз."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    dist = get_distance_func(metric)
    D = np.zeros((n, n))
    for i in range(n):
        D[i] = dist(X, X[i])
    return D


def compactness_profile(X, y, metric="manhattan"):
    """Π(m): доля ошибок 1NN при m первых эталонах."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(X)
    D = _pairwise_distances(X, metric)
    np.fill_diagonal(D, np.inf)

    ref_order = np.arange(n)
    errors = []
    
    for m in range(1, n + 1):
        active = ref_order[:m]
        sub = D[:, active]
        nearest_local = np.argmin(sub, axis=1)
        nearest_global = active[nearest_local]
        preds = y[nearest_global]
        
        for i in range(n):
            if nearest_global[i] == i:
                order = active[np.argsort(D[i, active])]
                second = next((j for j in order if j != i), None)
                if second is not None:
                    preds[i] = y[second]
                    
        err = np.mean(preds != y)
        errors.append(err)
    return np.array(errors)


def greedy_remove(X, y, metric="manhattan", max_iter=None):
    """Жадное удаление эталонов для 1NN (оптимизированная версия)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(X)
    D = _pairwise_distances(X, metric)
    np.fill_diagonal(D, np.inf)

    ref = np.ones(n, dtype=bool)
    if max_iter is None:
        max_iter = n

    def current_errors(ref_mask):
        active = np.where(ref_mask)[0]
        if len(active) < 2:
            return n
        
        # Берем только расстояния до активных эталонов
        sub = D[:, active]
        nearest_local = np.argmin(sub, axis=1)
        nearest_global = active[nearest_local]
        preds = y[nearest_global]
        
        # Если ближайший — это сам объект, берем следующего
        for i in active:
            if nearest_global[i] == i:
                order = np.argsort(sub[i])
                for pos in order:
                    j = active[pos]
                    if j != i:
                        preds[i] = y[j]
                        break
        return int(np.sum(preds != y))

    history = []
    cur_err = current_errors(ref)
    history.append((int(ref.sum()), cur_err / max(ref.sum(), 1)))

    for _ in range(max_iter):
        if ref.sum() <= 1:
            break
            
        best_j = -1
        best_err = cur_err
        
        # Пробуем удалить каждый эталон
        for j in np.where(ref)[0]:
            trial = ref.copy()
            trial[j] = False
            err = current_errors(trial)
            
            # Если ошибка не растет — запоминаем как кандидата
            if err <= best_err:
                if err < best_err or best_j == -1:
                    best_err = err
                    best_j = j
                    
        if best_j == -1:
            break
            
        ref[best_j] = False
        cur_err = best_err
        history.append((int(ref.sum()), cur_err / max(ref.sum(), 1)))

    return np.where(ref)[0], history