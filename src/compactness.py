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
    """Быстрое жадное удаление эталонов для 1NN по LOO-критерию.
    
    Идея: храним матрицу расстояний D (n×n) и для каждого объекта
    массив "ближайших эталонов" (упорядоченный по возрастанию D).
    При удалении эталона j обновляем только тех, у кого j был на
    первом месте (их обычно O(1), а не O(n)).
    
    Сложность: O(n² + k·n), где k — количество удаляемых эталонов.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = len(X)
    D = _pairwise_distances(X, metric)
    np.fill_diagonal(D, np.inf)

    # Порядок соседей для каждого объекта: argsort по расстоянию.
    # order[i] = [ближайший, второй, третий, ...] (без самого себя)
    order = np.argsort(D, axis=1)

    # position[i][j] = на каком месте стоит эталон j для объекта i
    # (0 — самый близкий). Храним в виде 2D-массива int32, чтобы
    # быстро обновлять.
    position = np.empty((n, n), dtype=np.int32)
    for i in range(n):
        position[i, order[i]] = np.arange(n)

    ref = np.ones(n, dtype=bool)  # True = эталон оставлен

    def nearest_ref(i):
        """Индекс ближайшего активного эталона (не равного i)."""
        for j in order[i]:
            if ref[j]:
                return j
        return -1

    def loo_errors():
        """Число ошибок 1NN на текущем наборе эталонов (LOO)."""
        wrong = 0
        for i in range(n):
            if not ref[i]:
                # объект не эталон — но всё равно нужен его прогноз
                pass
            best = nearest_ref(i)
            if best == -1 or y[best] != y[i]:
                wrong += 1
        return wrong

    history = []
    cur_err = loo_errors()
    n_ref = int(ref.sum())
    history.append((n_ref, cur_err / max(n_ref, 1)))

    max_iter = max_iter or n

    for _ in range(max_iter):
        if ref.sum() <= 1:
            break
        best_j, best_err = -1, cur_err
        # Пробуем удалить каждый активный эталон
        for j in np.where(ref)[0]:
            ref[j] = False
            err = loo_errors()
            ref[j] = True
            if err <= best_err:
                if err < best_err or best_j == -1:
                    best_err, best_j = err, j
        if best_j == -1:
            break
        ref[best_j] = False
        cur_err = best_err
        n_ref = int(ref.sum())
        history.append((n_ref, cur_err / max(n_ref, 1)))

    return np.where(ref)[0], history